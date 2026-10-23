# -*- coding: utf-8 -*-
"""
姿态解算实验（终版）—— 验证"终态倾角"判据
==========================================================
目的：
  验证创新点 A 的核心假设——
  「跌倒」与「坐下」在冲击峰值上不可分，但终态躯干倾角截然不同：
      · 跌倒   → 终态倾角大（水平倒地，>60°）且持续
      · 坐下   → 终态倾角小（竖直坐姿，<30°）

方法（关键工程决策）：
  终态倾角是在"人静止不动"时读取的 → 属于静态测量。
  静态下直接用 0.5Hz 低通加速度方向估算倾角：无需陀螺仪融合、
  无零偏漂移问题、腕表端实现简单。这是本实验验证后的选型结论。
  （Madgwick 六轴融合在合成数据上验证正确，但 SisFall 的 ITG3200
  存在高达 ~30°/s 的单轴零偏且无法离线重标定，长时间积分漂移严重，
  因此不用于终态判断——详见实验结论.md）

倾角定义：
  θ(t) = 当前低通加速度方向 与 起始站立参考方向 的夹角。
  该定义自动适应任意佩戴朝向，只反映"躯干从站立倒了多少度"，
  且对偏航不敏感。起始 1 秒作为站立参考段。

数据：SisFall，SA01-SA05，200Hz
  跌倒 F01-F15（375 文件）｜坐下 D07-D12｜躺下 D13-D14｜弯腰 D15-D16

输出：姿态倾角_跌倒vs坐下.png / 姿态特征_逐文件.csv
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
SUBJECTS = ["SA01", "SA02", "SA03", "SA04", "SA05"]
OUT = os.path.dirname(os.path.abspath(__file__))
FS = 200.0
ADXL_LSB2G = (2 * 16) / (2 ** 13)
ITG_LSB2DPS = (2 * 2000) / (2 ** 16)

GROUPS = {
    "跌倒": [f"F{i:02d}" for i in range(1, 16)],
    "坐下": [f"D{i:02d}" for i in range(7, 13)],
    "躺下": ["D13", "D14"],
    "弯腰": ["D15", "D16"],
}
LABEL2GROUP = {c: g for g, codes in GROUPS.items() for c in codes}

# ---------------------------------------------------------------- 数据与特征
def load_file(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.replace(";", "").split(",")
            if len(parts) >= 9:
                try:
                    rows.append([float(p) for p in parts[:9]])
                except ValueError:
                    continue
    d = np.asarray(rows)
    return d[:, 0:3] * ADXL_LSB2G, d[:, 3:6] * ITG_LSB2DPS

def tilt_accel(acc):
    """静态倾角（度）：0.5Hz 低通加速度方向 相对 起始站立方向 的夹角"""
    b, a = butter(2, 0.5, btype="low", fs=FS)
    sm = np.stack([filtfilt(b, a, acc[:, k]) for k in range(3)], axis=1)
    sm = sm / np.linalg.norm(sm, axis=1, keepdims=True)
    ref = sm[:int(FS)].mean(axis=0)
    ref /= np.linalg.norm(ref)
    return np.degrees(np.arccos(np.clip(sm @ ref, -1, 1)))

def sv_signal(acc):
    sv = np.sqrt((acc ** 2).sum(axis=1))
    b, a = butter(4, [0.5, 10], btype="band", fs=FS)
    return sv, filtfilt(b, a, sv)

# ---------------------------------------------------------------- 主流程
rows, examples = [], {}
for sub in SUBJECTS:
    folder = os.path.join(ROOT, sub)
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        code = fname.split("_")[0]
        if code not in LABEL2GROUP:
            continue
        try:
            acc, gyr = load_file(os.path.join(folder, fname))
            if len(acc) < 5 * int(FS):
                continue
            tilt = tilt_accel(acc)
            sv, sv_f = sv_signal(acc)
        except Exception as e:
            print("跳过", fname, e)
            continue

        event = int(np.argmax(sv_f))
        s1 = min(len(tilt) - 1, event + 3 * int(FS))
        e1 = min(len(tilt), event + 8 * int(FS))
        if e1 - s1 < int(FS):
            s1, e1 = min(len(tilt) - 1, event + int(FS)), len(tilt)
        seg = tilt[s1:e1]
        rows.append({
            "文件": fname, "受试者": sub, "动作组": LABEL2GROUP[code], "动作码": code,
            "终态倾角_中位": float(np.median(seg)),
            "终态倾角_最大": float(np.max(tilt[event:])),
            "倾角超60度时长_s": float((tilt[event:] > 60).sum() / FS),
            "事件后活动度": float(np.std(sv_f[s1:e1])),
        })
        if fname in ("F02_SA01_R01.txt", "D07_SA01_R01.txt", "F06_SA01_R01.txt", "D13_SA01_R01.txt"):
            t = (np.arange(len(tilt)) - event) / FS
            examples[fname] = (t, tilt, sv_f)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "姿态特征_逐文件.csv"), index=False, encoding="utf-8-sig")
print(f"完成 {len(df)} 个文件")
print(df.groupby("动作组")["终态倾角_中位"].describe()[["count", "mean", "50%", "min", "max"]].round(1))

falls = df[df.动作组 == "跌倒"]
other = df[df.动作组 != "跌倒"]
print("\n--- 判定规则扫描（终态倾角阈值 × 持续时长）---")
for th, dur in [(60, 3), (60, 2), (60, 1), (50, 3), (70, 3)]:
    rec = ((falls.终态倾角_中位 > th) & (falls.倾角超60度时长_s >= dur)).mean() * 100
    fa = ((other.终态倾角_中位 > th) & (other.倾角超60度时长_s >= dur)).mean() * 100
    print(f"  规则[倾角>{th}° 且 >60°持续≥{dur}s] → 跌倒召回 {rec:5.1f}% ｜ 日常误报 {fa:5.1f}%")

# ---------------------------------------------------------------- 出图
fig = plt.figure(figsize=(19, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1.15])

# (a) 示例曲线
ax = fig.add_subplot(gs[0, 0])
styles = {
    "F02_SA01_R01.txt": ("快速跌倒 F02（行走中后倒）", "#E24B4A"),
    "F06_SA01_R01.txt": ("晕厥垂直跌倒 F06", "#993C1D"),
    "D07_SA01_R01.txt": ("缓慢坐下起身 D07", "#378ADD"),
    "D13_SA01_R01.txt": ("躺下午睡 D13", "#639922"),
}
for fname, (t, tilt, sv_f) in examples.items():
    lab, c = styles[fname]
    ax.plot(t, tilt, color=c, lw=2, label=lab)
ax.axhline(60, color="#888780", ls="--", lw=1)
ax.text(-9, 63, "倒地判据 60°", fontsize=9, color="#5F5E5A")
ax.axhline(30, color="#B4B2A9", ls=":", lw=1)
ax.text(-9, 22, "竖直参考 30°", fontsize=9, color="#888780")
ax.set_xlim(-10, 10); ax.set_ylim(-5, 115)
ax.set_xlabel("相对事件的时间（秒）")
ax.set_ylabel("躯干倾角 θ（度）")
ax.set_title("(a) 倾角曲线：跌倒后长期 >60°；坐下/起身回到 <30°；\n躺下午睡是唯一同样 >60° 的日常动作", fontsize=11.5)
ax.legend(fontsize=8.5, loc="lower right")

# (b) 四组动作终态倾角箱线图
ax = fig.add_subplot(gs[0, 1])
order = ["跌倒", "坐下", "躺下", "弯腰"]
data = [df[df.动作组 == g].终态倾角_中位.values for g in order]
bp = ax.boxplot(data, patch_artist=True, flierprops=dict(markersize=3))
for patch, c in zip(bp["boxes"], ["#E24B4A", "#378ADD", "#639922", "#B77E14"]):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.set_xticklabels(order)
ax.axhline(60, color="#888780", ls="--", lw=1)
ax.set_ylabel("事件后 3-8 秒的倾角中位数（度）")
ax.set_title("(b) 终态倾角分布：跌倒 vs 坐下/弯腰 明显分离，\n与躺下重叠（需时长/语音确认区分）", fontsize=11.5)

# (c) 二维特征散点
ax = fig.add_subplot(gs[0, 2])
for g, c in [("跌倒", "#E24B4A"), ("坐下", "#378ADD"), ("躺下", "#639922"), ("弯腰", "#B77E14")]:
    d = df[df.动作组 == g]
    ax.scatter(d.终态倾角_中位, d.倾角超60度时长_s, s=26, alpha=0.65, color=c, label=g)
ax.axvline(60, color="#888780", ls="--", lw=1)
ax.axhline(3, color="#888780", ls="--", lw=1)
ax.fill_between([60, 115], 3, 14, color="#E24B4A", alpha=0.06)
ax.text(64, 11.5, "判定为跌倒区\n（倾角>60° 且持续≥3s）", fontsize=9.5, color="#A32D2D")
ax.set_xlim(0, 115); ax.set_ylim(-0.3, 14)
ax.set_xlabel("终态倾角中位数（度）")
ax.set_ylabel("倾角 >60° 的持续时长（秒）")
ax.set_title("(c) 双特征联合判定", fontsize=11.5)
ax.legend(fontsize=9, loc="upper left")

fig.suptitle("姿态解算实验（SisFall · SA01-SA05 · 625 文件）：验证终态倾角判据", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, "姿态倾角_跌倒vs坐下.png"), dpi=150)
print("图已保存")

print("\n终态倾角最小的 8 个跌倒样本（漏报风险）：")
print(falls.nsmallest(8, "终态倾角_中位")[["文件", "动作码", "终态倾角_中位", "倾角超60度时长_s"]].round(1).to_string(index=False))
print("\n落入判定区的非跌倒样本（误报风险）：")
bad = other[(other.终态倾角_中位 > 60) & (other.倾角超60度时长_s >= 3)]
print(bad[["文件", "动作组", "动作码", "终态倾角_中位", "倾角超60度时长_s"]].round(1).to_string(index=False))
