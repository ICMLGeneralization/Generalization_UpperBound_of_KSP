import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["text.latex.preamble"] = r"""
\usepackage{newtxtext}
\usepackage{newtxmath}
"""
import os

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "ETTh_UpperBound_Visualization.csv"))

x = df["x"].values
theo1 = df["Transformer"].values
expr1_mean = df["expr1_mean"].values
expr1_lower = df["expr1_lower"].values
expr1_upper = df["expr1_upper"].values
theo2 = df["Autoformer"].values
expr2_mean = df["expr2_mean"].values
expr2_lower = df["expr2_lower"].values
expr2_upper = df["expr2_upper"].values
theo3 = df["PatchTST"].values
expr3_mean = df["expr3_mean"].values
expr3_lower = df["expr3_lower"].values
expr3_upper = df["expr3_upper"].values
theo4 = df["DLinear"].values
expr4_mean = df["expr4_mean"].values
expr4_lower = df["expr4_lower"].values
expr4_upper = df["expr4_upper"].values


colors = ["#FF5500", "#FFD400", "#40FF00", "#246ADB"]
colors2 = ["#AB2B38", "#D68B00", "#267E31", "#333870"]

line_theo1, = plt.plot(
    x, theo1, linestyle="--", color=colors[0], label="[Theoretical] Transformer"
)
line_expr1, = plt.plot(
    x, expr1_mean, linestyle="-", color=colors2[0], label="[Experimental] Transformer"
)
line_theo2, = plt.plot(
    x, theo2, linestyle="--", color=colors[1], label="[Theoretical] Autoformer"
)
line_expr2, = plt.plot(
    x, expr2_mean, linestyle="-", color=colors2[1], label="[Experimental] Autoformer"
)
line_theo3, = plt.plot(
    x, theo3, linestyle="--", color=colors[2], label="[Theoretical] PatchTST"
)
line_expr3, = plt.plot(
    x, expr3_mean, linestyle="-", color=colors2[2], label="[Experimental] PatchTST"
)
line_theo4, = plt.plot(
    x, theo4, linestyle="--", color=colors[3], label="[Theoretical] DLinear"
)
line_expr4, = plt.plot(
    x, expr4_mean, linestyle="-", color=colors2[3], label="[Experimental] DLinear"
)

for line, lower, upper in [
    (line_expr1, expr1_lower, expr1_upper),
    (line_expr2, expr2_lower, expr2_upper),
    (line_expr3, expr3_lower, expr3_upper),
    (line_expr4, expr4_lower, expr4_upper),
]:
    color = line.get_color()
    plt.fill_between(x, lower, upper, color=color, alpha=0.5)

plt.xlabel(r"$T$", fontsize=14)
plt.ylabel(r"Upper Bound", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("figure.png", dpi=300)

