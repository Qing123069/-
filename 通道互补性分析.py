# -*- coding: utf-8 -*-
"""
通道互补性与标注口径敏感性分析
==========================================================
回答三个可能被评委追问的问题：

Q1（标注口径）"F16-19 官方标注是跌倒，你们凭什么算日常？把它们算回去，召回还剩多少？"
   → 事实核查：SisFall 跌倒只有 F01-F15（全库代码枚举为证），"坐/起时向前后侧倒"= F08-F12，
     本来就是跌倒、我们按跌倒统计。真正"类跌倒"的是 ADL 中的 D11（起身失败瘫回椅子）
     与 D18（行走被绊未倒）——官方标注即 ADL，不存在改标。
   → 但仍然给出【敏感性与口径表】：若评审采用更严格口径（把 D11/D18 这类"类跌倒日常动作"
     也视为应当报警），召回会变成多少？让评审可以按自己的口径复核。

Q2（通道互补性）"姿态通道漏掉的跌倒，谁来接住？"
   → 统计姿态通道漏报的跌倒文件的冲击峰值分布，看冲击通道能否独立触发。

Q3（双通道必要性）"只留冲击通道行不行？"
   → 统计类跌倒日常动作（D11/D18）的冲击峰值——若显著高于普通日常，说明纯冲击通道会误报，
     必须由姿态通道 + 语音确认兜底。这是消融证据。

峰值口径与 02_SisFall数据分析 完全一致：SV 带通滤波（0.5-10 Hz）后取最大值。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"E:\防摔倒检测\SisFall\SisFall_dataset\SisFall_dataset"
OUT = os.path.dirname(os.path.abspath(__file__))
FEAT = r"C:\Users\lenovo\WorkBuddy\2026-09-24-19-01-14\防跌倒项目\08_全量复验\逐文件特征_全量.csv"
FS = 200.0
ADXL_LSB2G = (2 * 16) / (2 ** 13)
TH, DUR = 60.0, 3.0
NEAR_FALL_ADL = ["D11", "D18"]          # 官方标注 ADL 的"类跌倒"动作
SEATED_END_FALLS = ["F13", "F14", "F15"]  # 坐姿晕厥终结的跌倒（官方=跌倒）

def load_acc(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read().replace(";", " ").replace(",", " ")
    arr = np.array(s.split(), dtype=float)
    if arr.size < 9:
        return None
    d = arr.reshape(-1, 9)
    d = d[np.isfinite(d).all(axis=1)]
    return d[:, 0:3] * ADXL_LSB2G

def sv_peaks(acc):
    sv = np.sqrt((acc ** 2).sum(axis=1))
    b, a = butter(4, [0.5, 10], btype="band", fs=FS)
    sv_f = filtfilt(b, a, sv)
    return float(sv.max()), float(sv_f.max())

# ---------------------------------------------------------------- 逐文件峰值
rows = []
for sub in sorted(os.listdir(ROOT)):
    folder = os.path.join(ROOT, sub)
    if not os.path.isdir(folder):
        continue
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        acc = load_acc(os.path.join(folder, fname))
        if acc is None:
            continue
        raw, filt = sv_peaks(acc)
        rows.append({"文件": fname, "受试者": sub, "动作码": fname.split("_")[0],
                     "标签": "跌倒" if fname.startswith("F") else "日常",
                     "SV峰值_原始g": raw, "SV峰值_滤波g": filt})

peak = pd.DataFrame(rows)
feat = pd.read_csv(FEAT)
# 注意：SisFall 打包瑕疵——D17_SE15_R01~R05 同时存在于 SA15 与 SE15 目录（命名均为 SE15），
# 因此必须按"受试者 + 文件名"联合键合并，否则会被放大成重复样本。
df = feat.merge(peak[["文件", "受试者", "SV峰值_原始g", "SV峰值_滤波g"]], on=["受试者", "文件"], how="left")
assert df.SV峰值_滤波g.notna().all(), "存在未匹配到峰值的文件"
assert len(df) == len(feat), f"合并后条数异常：{len(df)} vs {len(feat)}"
df["姿态通道触发"] = (df.终态倾角_中位 > TH) & (df.倾角超60度时长_s >= DUR)
df.to_csv(os.path.join(OUT, "通道互补性_逐文件.csv"), index=False, encoding="utf-8-sig")
print(f"逐文件峰值计算完成：{len(df)} 条（跌倒 {len(df[df.标签=='跌倒'])} / 日常 {len(df[df.标签=='日常'])}）")

falls = df[df.标签 == "跌倒"]
adls = df[df.标签 == "日常"]
missed = falls[~falls.姿态通道触发]          # 姿态通道漏报的跌倒
near = adls[adls.动作码.isin(NEAR_FALL_ADL)]  # 类跌倒日常动作
plain = adls[~adls.动作码.isin(NEAR_FALL_ADL)]

# ---------------------------------------------------------------- Q2 互补性
print("\n===== Q2: 姿态通道漏报的跌倒，冲击通道能否接住 =====")
print(f"姿态通道漏报跌倒 {len(missed)} 例 / 全部 {len(falls)} 例")
for t in (1.38, 2.0, 2.5, 3.0):
    n = int((missed.SV峰值_滤波g > t).sum())
    print(f"  其中冲击峰值 > {t} g 的: {n} 例（{n/len(missed)*100:.1f}%）")
print(f"  漏报样本冲击峰值：中位 {missed.SV峰值_滤波g.median():.2f} g，"
      f"最小 {missed.SV峰值_滤波g.min():.2f} g，最大 {missed.SV峰值_滤波g.max():.2f} g")
print(f"  对照：全部跌倒冲击峰值中位 {falls.SV峰值_滤波g.median():.2f} g")
print("\n  漏报样本明细（按峰值升序）：")
print(missed.nsmallest(len(missed), "SV峰值_滤波g")[
    ["文件", "动作码", "动作", "终态倾角_中位", "SV峰值_滤波g"]].round(2).to_string(index=False))

# ---------------------------------------------------------------- Q3 双通道必要性
print("\n===== Q3: 只留冲击通道会怎样（类跌倒日常动作的峰值）=====")
print(f"类跌倒日常动作 D11/D18：{len(near)} 例，冲击峰值中位 {near.SV峰值_滤波g.median():.2f} g")
for t in (1.38, 2.0, 2.5):
    n = int((near.SV峰值_滤波g > t).sum())
    print(f"  冲击峰值 > {t} g 的: {n} 例（{n/len(near)*100:.1f}%）")
print(f"普通日常动作（其余 17 类）：{len(plain)} 例，冲击峰值中位 {plain.SV峰值_滤波g.median():.2f} g")
for t in (1.38, 2.0):
    n = int((plain.SV峰值_滤波g > t).sum())
    print(f"  冲击峰值 > {t} g 的: {n} 例（{n/len(plain)*100:.1f}%）")
print(f"\n全部日常动作冲击峰值中位 {adls.SV峰值_滤波g.median():.2f} g")

# ---------------------------------------------------------------- Q1 口径敏感性
print("\n===== Q1: 标注口径敏感性（官方口径 vs 严格口径）=====")
tp = int(falls.姿态通道触发.sum()); fn = int(len(falls) - tp)
fp = int(adls.姿态通道触发.sum()); tn = int(len(adls) - fp)

def make_row(name, extra_pos, neg_df, neg_trig):
    """extra_pos: 额外计入"应报警"的样本数（严格口径下把类跌倒日常算作应报警）"""
    pos_total = tp + fn + extra_pos
    rec = tp / pos_total if pos_total else 0
    fa = neg_trig / len(neg_df) if len(neg_df) else 0
    return dict(口径=name, 应报警样本数=pos_total,
                召回率=f"{rec*100:.2f}%", 日常样本数=len(neg_df),
                误报数=neg_trig, 误报率=f"{fa*100:.2f}%")

plain_trig = int(plain.姿态通道触发.sum())
seated = falls[falls.动作码.isin(SEATED_END_FALLS)]
rows_out = [
    make_row("A 官方口径（F01-F15=跌倒，D01-D19=日常）", 0, adls, fp),
    make_row("B 严格口径（额外把类跌倒日常 D11/D18 视为应当报警）", len(near), plain, plain_trig),
    dict(口径="C 官方口径 · 单看坐姿终结跌倒 F13-F15（官方即跌倒）",
         应报警样本数=len(seated), 召回率=f"{seated.姿态通道触发.mean()*100:.2f}%",
         日常样本数="—", 误报数="—", 误报率="—"),
    dict(口径="D 官方口径 · 单看类跌倒日常 D11/D18（官方即日常）",
         应报警样本数="—", 召回率="—",
         日常样本数=len(near), 误报数=int(near.姿态通道触发.sum()),
         误报率=f"{near.姿态通道触发.mean()*100:.2f}%"),
]
for r in rows_out:
    print(f"  {r['口径']}")
    print(f"      应报警 {r['应报警样本数']} ｜ 召回 {r['召回率']} ｜ 日常 {r['日常样本数']} ｜ 误报 {r['误报数']}（{r['误报率']}）")
sens = pd.DataFrame(rows_out)
sens.to_csv(os.path.join(OUT, "口径敏感性表.csv"), index=False, encoding="utf-8-sig")
print("\n  口径表已保存")

# ---------------------------------------------------------------- 图
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
groups = [
    ("姿态漏报的跌倒\n(n=%d)" % len(missed), missed.SV峰值_滤波g.values, "#993C1D"),
    ("全部跌倒\n(n=%d)" % len(falls), falls.SV峰值_滤波g.values, "#E24B4A"),
    ("类跌倒日常 D11/D18\n(n=%d)" % len(near), near.SV峰值_滤波g.values, "#B77E14"),
    ("普通日常\n(n=%d)" % len(plain), plain.SV峰值_滤波g.values, "#378ADD"),
]
bp = ax.boxplot([g[1] for g in groups], patch_artist=True, showfliers=True, flierprops=dict(markersize=3))
for p, g in zip(bp["boxes"], groups):
    p.set_facecolor(g[2]); p.set_alpha(0.6)
ax.set_xticklabels([g[0] for g in groups], fontsize=9)
for t, ls in [(1.38, "--"), (2.5, ":")]:
    ax.axhline(t, color="#888780", ls=ls, lw=1)
    ax.text(4.55, t + 0.08, f"{t} g", fontsize=8, color="#5F5E5A")
ax.set_ylabel("冲击峰值（滤波后 SV 最大值, g）")
ax.set_title("(a) 冲击通道视角：姿态漏报的跌倒仍有高冲击；\n而类跌倒日常动作同样有高冲击 → 单通道必失", fontsize=11)

ax = axes[1]
ax.scatter(falls.SV峰值_滤波g, falls.终态倾角_中位, s=26, alpha=0.6, color="#E24B4A", label="跌倒 F01-F15")
ax.scatter(near.SV峰值_滤波g, near.终态倾角_中位, s=30, alpha=0.8, color="#B77E14", marker="^", label="类跌倒日常 D11/D18")
ax.scatter(plain.SV峰值_滤波g, plain.终态倾角_中位, s=16, alpha=0.35, color="#378ADD", label="普通日常")
ax.axvline(1.38, color="#888780", ls="--", lw=1); ax.axhline(60, color="#888780", ls="--", lw=1)
ax.fill_between([1.38, 8], 3, 120, color="#3B6D11", alpha=0.05)
ax.text(1.5, 108, "姿态通道判定区\n(倾角>60° 且 ≥3s)", fontsize=9, color="#3B6D11")
ax.set_xlim(0, 7.5); ax.set_ylim(-3, 120)
ax.set_xlabel("冲击峰值（g）"); ax.set_ylabel("终态佩戴部位倾角（度）")
ax.set_title("(b) 双通道互补：右下=冲击通道兜底（姿态漏报）；\n左侧=冲击通道失效区（慢速跌倒，靠姿态通道）", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
fig.suptitle("通道互补性与口径敏感性（SisFall 全量 4505 条）", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, "通道互补性_全量.png"), dpi=150)
print("图已保存: 通道互补性_全量.png")
