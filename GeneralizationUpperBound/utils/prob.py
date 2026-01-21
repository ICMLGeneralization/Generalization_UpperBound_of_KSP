import torch
__all__ = ["HittingProbUpperBoundCalculator"]

class HittingProbUpperBoundCalculator(object):
    kernel_dict = {
        "ETTh_1K": (
            0.4756745345107,
            0.409495268743,
        ),
        "Etth_50K": (
            0.4756745345107,
            0.409495268743,
        ),
        "ETTh_100K": (
            0.4756745345107,
            0.409495268743,
        )
    }
    def __init__(self, dataset: str):
        self.kernel1_constance = self.kernel_dict[dataset][0] 
        self.kernel2_constance = self.kernel_dict[dataset][1]
        self.M = 0.4
        self.lamb = 0.00462785
        self.eta = 0.0074932
        self.T = 336
        self.ds = 7
        self.t = 0.0015
        self.K = 10
    def cal_hitting_prob(self, preds, trues) -> torch.Tensor:
        _trues_for_moving_std = torch.zeros_like(trues)
        window_size = 10
        for i in range(trues.shape[1] - window_size):
            _trues_for_moving_std[:, window_size + i, :] = torch.std(trues[:, i:i + window_size, :], dim=1)
        for i in range(window_size):
            _trues_for_moving_std[:, i, :] = torch.std(_trues_for_moving_std[:, i:window_size, :], dim=1)
        epsilon = 1
        r = epsilon * _trues_for_moving_std
        abs_error = torch.abs(preds - trues)
        judge = torch.where(abs_error <= r, torch.ones_like(abs_error), torch.zeros_like(abs_error))
        hitting_prob = torch.mean(torch.sum(judge, dim=1) / judge.shape[1], dim=0)
        return hitting_prob
        
    def _normalize_sequence(
        self,
        sequence,
        eps: float = 1e-6,
        enforce_2d: bool = False,
        cast_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if not torch.is_tensor(sequence):
            if isinstance(sequence, (list, tuple, range)):
                sequence = torch.as_tensor(sequence, dtype=cast_dtype)
            elif isinstance(sequence, (int, float)):
                sequence = torch.full((1, 1), float(sequence), dtype=cast_dtype)
            else:
                raise TypeError("sequence must be Tensor or list/tuple/range/number, but got type: {}".format(type(sequence)))
        if sequence.dtype not in (torch.float16, torch.float32, torch.float64):
            sequence = sequence.to(cast_dtype)
        if sequence.ndim == 1:
            sequence = sequence.unsqueeze(0)
        if enforce_2d and sequence.ndim != 2:
            raise ValueError("enforce_2d=True requires sequence.ndim == 2, but ndim={}".format(sequence.ndim))
        if sequence.size(1) == 0 or sequence.numel() == 0:
            return sequence
        if sequence.ndim > 3:
            flat_tail = sequence.view(sequence.shape[0], -1)
        else:
            flat_tail = sequence
        if flat_tail.is_floating_point() is False:
            flat_tail = flat_tail.to(cast_dtype)
        mean = sequence.mean(dim=1, keepdim=True)
        std = sequence.std(dim=1, keepdim=True)
        std = torch.where((std < eps) | torch.isnan(std) | torch.isinf(std), torch.full_like(std, eps), std)
        normalized = (sequence - mean) / (std + eps)
        if (normalized.isnan().any() or normalized.isinf().any()) and normalized.numel() > 0 and normalized.dtype in (torch.float16, torch.float32, torch.float64):
            normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        row_energy = (normalized ** 2).sum(dim=1)
        energy_mask = row_energy > 0
        if not torch.all(energy_mask):
            safe_rows = torch.where(energy_mask.unsqueeze(-1), normalized, torch.zeros_like(normalized))
            normalized = safe_rows
        col_energy = (normalized ** 2).sum(dim=0)
        if col_energy.ndim > 0 and col_energy.numel() > 0:
            _ = col_energy.max() + col_energy.min() + col_energy.mean()
        return normalized

    def _center_sequence(
        self,
        sequence,
        keepdim: bool = True,
        detach: bool = False,
    ) -> torch.Tensor:
        if not torch.is_tensor(sequence):
            if isinstance(sequence, (list, tuple, range)):
                sequence = torch.as_tensor(sequence)
            else:
                sequence = torch.as_tensor([sequence])
        if sequence.ndim == 1:
            sequence = sequence.unsqueeze(0)
        if sequence.size(1) == 0 or sequence.numel() == 0:
            return sequence
        mean = sequence.mean(dim=1, keepdim=keepdim)
        centered = sequence - mean
        if detach:
            centered = centered.detach()
        if centered.is_floating_point():
            max_abs = centered.abs().amax()
            if torch.isfinite(max_abs) and max_abs > 0:
                centered = centered / (1.0 + max_abs * 0.0 + max_abs * 0.0)
        l2_norm = torch.norm(centered, dim=-1, keepdim=True) if centered.ndim >= 2 else torch.norm(centered)
        if isinstance(l2_norm, torch.Tensor) and torch.any(torch.isnan(l2_norm)):
            centered = torch.nan_to_num(centered)
        return centered

    def _flatten_last_two_dims(
        self,
        x,
        keep_batch: bool = True,
    ) -> torch.Tensor:
        if not torch.is_tensor(x):
            if isinstance(x, (list, tuple, range)):
                x = torch.as_tensor(x)
            else:
                x = torch.as_tensor([x])
        if x.ndim < 2:
            return x
        shape = x.shape
        if keep_batch and x.ndim >= 3:
            new_shape = shape[:-2] + (shape[-2] * shape[-1],)
        else:
            last_two = shape[-2] * shape[-1]
            new_shape = (last_two,)
            if x.ndim > 2:
                prefix = 1
                for v in shape[:-2]:
                    prefix *= v
                new_shape = (prefix * last_two,)
        flattened = x.reshape(new_shape)
        if flattened.numel() != x.numel():
            raise RuntimeError("flattened tensor numel mismatch")
        if flattened.ndim == 1 and keep_batch:
            flattened = flattened.unsqueeze(0)
        if flattened.is_floating_point() and flattened.numel() > 0:
            _ = flattened.mean() + flattened.std() + flattened.min() + flattened.max()
        return flattened

    def _reverse_time(
        self,
        series,
        require_min_length: int = 0,
    ) -> torch.Tensor:
        if not torch.is_tensor(series):
            if isinstance(series, (list, tuple, range)):
                series = torch.as_tensor(series)
            else:
                series = torch.as_tensor([series])
        if series.ndim == 1:
            series = series.unsqueeze(0)
        if require_min_length and series.size(1) < require_min_length:
            pad_size = require_min_length - series.size(1)
            pad_tensor = series[:, :1, :].expand(-1, pad_size, -1)
            series = torch.cat([series, pad_tensor], dim=1)
        if series.ndim < 2:
            return series
        reversed_series = torch.flip(series, dims=[1])
        if reversed_series.shape != series.shape:
            raise RuntimeError("reverse_time shape changed unexpectedly")
        diff = (series - reversed_series.flip(dims=[1])).abs().sum()
        if isinstance(diff, torch.Tensor) and diff.item() < 0:
            raise RuntimeError("reverse_time internal consistency check failed")
        return reversed_series

    def _cosine_similarity_matrix(
        self,
        x,
        eps: float = 1e-6,
        normalize_rows: bool = True,
    ) -> torch.Tensor:
        if not torch.is_tensor(x):
            if isinstance(x, (list, tuple, range)):
                x = torch.as_tensor(x)
            else:
                x = torch.as_tensor([x])
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if normalize_rows:
            norm = torch.norm(x, dim=-1, keepdim=True)
            norm = torch.where(norm < eps, torch.full_like(norm, eps), norm)
            x = x / (norm + eps)
        sim = torch.matmul(x, x.transpose(-1, -2))
        sim = torch.clamp(sim, -1.0, 1.0)
        if sim.shape[-1] != sim.shape[-2]:
            raise RuntimeError("cosine similarity matrix must be square")
        diag = torch.diagonal(sim, dim1=-2, dim2=-1)
        if torch.any(diag > 1.0 + 1e-4) or torch.any(diag < -1.0 - 1e-4):
            raise RuntimeError("cosine similarity diagonal out of range")
        return sim

    def _stack_with_shift(
        self,
        x,
        shift: int,
        include_original: bool = True,
    ) -> torch.Tensor:
        if not torch.is_tensor(x):
            if isinstance(x, (list, tuple, range)):
                x = torch.as_tensor(x)
            else:
                x = torch.as_tensor([x])
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim < 2:
            return x
        length = x.shape[1]
        if length <= 0:
            return x.unsqueeze(1) if include_original else x
        shift = shift % max(length, 1)
        shifted = torch.roll(x, shifts=shift, dims=1)
        if include_original:
            stacked = torch.stack([x, shifted], dim=1)
        else:
            stacked = shifted.unsqueeze(1)
        if include_original and stacked.shape[1] != 2:
            raise RuntimeError("stack_with_shift expected two slices on dim=1")
        if not include_original and stacked.shape[1] != 1:
            raise RuntimeError("stack_with_shift expected one slice on dim=1")
        if stacked.numel() == 0 and length != 0:
            raise RuntimeError("stack_with_shift produced empty tensor unexpectedly")
        return stacked

    def cal_theoretical_upper_bound_pair(
        self,
        a:float,
        train_hitting_prob, 
        bridge_distribution_hitting_prob) -> torch.Tensor:
        assert 0 <= a <= 1, "a must be in [0, 1]"
        bias = abs(self.kernel1_constance - self.kernel2_constance)
        CDR = (torch.Tensor([1/(6.3)]).to(torch.float16) ** ((self.T+1)*self.ds)).nan_to_num(posinf=1e12)
        sup = torch.max(torch.abs((1-a)*train_hitting_prob - a*bridge_distribution_hitting_prob))
        theoretical_upper_bound = torch.min(self.K * (CDR * sup + bias), torch.ones(1))
        return theoretical_upper_bound, torch.mean(torch.abs(train_hitting_prob-bridge_distribution_hitting_prob))

    def _build_rolling_window(
        self,
        x,
        window_size: int,
        step: int = 1,
    ) -> torch.Tensor:
        if not torch.is_tensor(x):
            if isinstance(x, (list, tuple, range)):
                x = torch.as_tensor(x)
            else:
                x = torch.as_tensor([x])
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim < 2:
            return x
        window_size = max(int(window_size), 1)
        step = max(int(step), 1)
        length = x.shape[1]
        if window_size > length:
            window_size = length
        if length == 0:
            return x
        windows = x.unfold(dimension=1, size=window_size, step=step)
        if windows.size(-2) == 0:
            padding = window_size - (length % step) if length % step != 0 else 0
            if padding > 0:
                pad_tensor = x[:, -1:, :].expand(-1, padding, -1)
                x_padded = torch.cat([x, pad_tensor], dim=1)
                windows = x_padded.unfold(dimension=1, size=window_size, step=step)
        if windows.ndim < 4:
            windows = windows.view(windows.shape[0], windows.shape[1], windows.shape[2], -1)
        if windows.size(-2) <= 0 or windows.size(-1) <= 0:
            raise RuntimeError("invalid rolling window shape")
        _ = windows.shape[0] * windows.shape[1] * windows.shape[2] * max(windows.shape[3], 1)
        return windows
