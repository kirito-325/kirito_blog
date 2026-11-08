"""Generate the article's schematic figures; no model or experimental data is used.

Run: python3 tools/generate_openjev_rlcd_figures.py
Requires matplotlib and numpy. OPENJEV_FIGURE_FONT can point to a CJK font.
"""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "openjev-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/img/posts/openjev-rlcd"
BLUE = "#2364AA"
ORANGE = "#C06A28"
GREEN = "#27806C"
INK = "#213247"
GRAY = "#64748B"
GRID = "#E6EBF0"


def configure():
    candidates = [
        os.environ.get("OPENJEV_FIGURE_FONT", ""),
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if path and Path(path).is_file():
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            break
    else:
        raise RuntimeError("Set OPENJEV_FIGURE_FONT to a font supporting Chinese.")
    plt.rcParams.update({
        "font.family": family,
        "font.size": 11.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11.5,
        "axes.edgecolor": "#B4C0CD",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
    })
    OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    path = OUT / (name + ".png")
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(path.relative_to(ROOT))


def calibration():
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    fig.subplots_adjust(bottom=0.19, top=0.88, left=0.13, right=0.96)
    c = np.array([.55, .65, .75, .85, .95])
    ax.plot([.45, 1], [.45, 1], color="#A1ACB9", lw=1.3, ls="--", zorder=1)
    ax.plot(c, c, "o-", color=BLUE, lw=2.1, ms=6, label="接近校准：正确率与置信度相符")
    ax.plot(c, [.40, .49, .59, .68, .78], "s-", color=ORANGE, lw=1.9, ms=6,
            label="过度自信：实际正确率更低")
    ax.plot(c, [.68, .77, .85, .93, .99], "^-", color=GREEN, lw=1.9, ms=6,
            label="偏于保守：实际正确率更高")
    ax.set(xlim=(.48, 1), ylim=(.30, 1.04),
           xlabel="同一组预测的平均置信度", ylabel="这一组预测的实际正确率")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks([.5, .6, .7, .8, .9, 1])
    ax.grid(color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    fig.suptitle("校准要看一组预测：报出的把握，能否对应实际表现", fontsize=14, y=.98)
    fig.text(.5, .035, "示意数据，非论文实验结果；每个点代表一组相近置信度的预测。",
             ha="center", color=GRAY, fontsize=10)
    save(fig, "calibration-reliability")


def cross_entropy():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
    fig.subplots_adjust(left=.075, right=.98, bottom=.22, top=.76, wspace=.29)
    p = np.linspace(.01, .99, 600)
    ce = -.4 * np.log2(p) - .6 * np.log2(1 - p)
    hp = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    hq = -.4 * np.log2(.4) - .6 * np.log2(.6)
    ce_at_01 = -.4 * np.log2(.1) - .6 * np.log2(.9)
    hp_at_01 = -.1 * np.log2(.1) - .9 * np.log2(.9)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"预测 A 获胜的概率 $p_A$")
        ax.set_ylabel("bit")
        ax.set_xticks([0, .2, .4, .6, .8, 1])
        ax.grid(color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        ax.axvline(.4, color="#BAC4D0", lw=1.1, ls="--")
    axes[0].plot(p, ce, color=BLUE, lw=2.5)
    axes[0].scatter([.1, .4], [ce_at_01, hq], color=BLUE, s=45, zorder=3)
    axes[0].set_ylim(.7, 4.2)
    axes[0].set_title(r"交叉熵 $H(q,p)$：关于 $p$ 是凸的", pad=14)
    axes[0].annotate("最低点\np = q", xy=(.4, hq), xytext=(.55, 1.75),
                     arrowprops={"arrowstyle": "->", "color": GRAY}, fontsize=11)
    axes[0].text(.035, 2.05, "p = (0.1, 0.9)\nH(q,p) = 1.420", fontsize=10.5)
    axes[0].text(.5, -.32, "从 pA = 0.1 调整到 0.4：交叉熵降低", ha="center",
                 transform=axes[0].transAxes, color=BLUE, fontsize=10.5)
    axes[1].plot(p, hp, color=ORANGE, lw=2.5)
    axes[1].scatter([.1, .4], [hp_at_01, hq], color=ORANGE, s=45, zorder=3)
    axes[1].set_ylim(0, 1.17)
    axes[1].set_title(r"预测分布的熵 $H(p)$：关于 $p$ 是凹的", pad=14)
    axes[1].annotate("此时预测熵反而升高", xy=(.4, hq), xytext=(.43, .65),
                     arrowprops={"arrowstyle": "->", "color": GRAY}, fontsize=10.5)
    axes[1].text(.5, -.32, "同一次调整：预测熵从 0.469 升至 0.971", ha="center",
                 transform=axes[1].transAxes, color=ORANGE, fontsize=10.5)
    fig.suptitle("降低交叉熵，与降低预测分布自身的熵，是两件事", y=.98, fontsize=14)
    fig.text(.5, .86, "固定真实分布 q = (0.4, 0.6)；两个图都改变同一个预测概率 pA。",
             ha="center", fontsize=11, color=GRAY)
    save(fig, "cross-entropy-and-entropy")


def token_readout():
    # The first four cells are individual option tokens. The last cell aggregates
    # all unshown tokens. These invented masses sum to one, just as a full softmax does.
    masses = np.array([
        .08, .16, .04, .02,
        .03, .025, .065, .045,
        .055, .035, .04, .025,
        .025, .02, .015, .32,
    ])
    assert np.isclose(masses.sum(), 1)
    labels = [" A", " B", " C", " D", "yes", "no", "The", "I",
              "\\n", "answer", "It", "because", ".", ",", "<eos>", "其余合计"]
    options = masses[:4] / masses[:4].sum()
    fig = plt.figure(figsize=(10.2, 5.7))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, .92], left=.035, right=.96,
                           bottom=.20, top=.76, wspace=.40)
    ax = fig.add_subplot(grid[0])
    bars = fig.add_subplot(grid[1])
    ax.imshow(masses.reshape(4, 4), cmap="Blues", vmin=0, vmax=.32)
    for i, (label, value) in enumerate(zip(labels, masses)):
        row, col = divmod(i, 4)
        color = "white" if value >= .16 else INK
        ax.text(col, row, f"{label}\n{value:.1%}", ha="center", va="center", color=color, fontsize=11)
    for col in range(4):
        ax.add_patch(Rectangle((col - .49, -.49), .98, .98, fill=False,
                               edgecolor=BLUE, linewidth=1.4))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("完整词表概率的示意切片", pad=13)
    bars.barh(["A", "B", "C", "D"], options, color=BLUE, height=.58)
    bars.invert_yaxis()
    bars.set_xlim(0, .68)
    bars.set_xlabel("在候选选项内重新归一化的概率")
    bars.xaxis.set_major_formatter(PercentFormatter(1))
    bars.set_xticks([0, .2, .4, .6])
    bars.grid(axis="x", color=GRID, lw=.8)
    bars.set_axisbelow(True)
    bars.set_title("只在 A / B / C / D 之间比较", pad=13)
    for i, value in enumerate(options):
        bars.text(value + .018, i, f"{value:.1%}", va="center", fontsize=11)
    fig.suptitle("从下一 token 的词表分布，读取候选答案的分布", fontsize=14, y=.98)
    fig.text(.5, .85, "选项 token 的原始概率总和为 30%；选项内归一化后，总和为 100%。",
             ha="center", fontsize=11, color=GRAY)
    fig.text(.5, .055, "全部数值均为示意；“其余合计”代表未展示 token 的总概率。",
             ha="center", color=GRAY, fontsize=10.5)
    save(fig, "next-token-to-options")


if __name__ == "__main__":
    configure()
    calibration()
    cross_entropy()
    token_readout()
