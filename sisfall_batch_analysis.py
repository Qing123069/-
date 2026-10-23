# -*- coding: utf-8 -*-
"""
SisFall 批量分析：跌倒(F01-F15) vs 日常活动(D01-D19) 的合加速度峰值对比
- SA01-SA05 共 5 名受试者，全部 F 文件 + 全部 D 文件
- 传感器：第 1-3 列 ADXL345 加速度（±16g, 13bit），换算为 g
- 输出：峰值对比直方图 + 每类动作箱线图 + 阈值法性能曲线 + 每文件统计 CSV
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
FS = 200
ADXL_LSB2G = (2 * 16) / (2 ** 13)   # 0.00390625 g/bit

def _parse_lines(path):
    """逐行解析，跳过格式异常/空行（个别文件末尾有多余空行）"""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.replace(";", "").split(",")
            if len(parts) >= 9:
                try:
                    rows.append([float(p) for p in parts[:9]])
                except ValueError:
                    continue
    return np.asarray(rows)

def load_sv(path):
    """读取一个文件，返回（原始SV[g], 滤波后SV[g]）"""
    try:
        data = np.loadtxt(path, delimiter=",",
                          converters=lambda s: float(str(s).replace(";", "").strip()))
        if data.ndim != 2 or data.shape[1] != 9:
            raise ValueError
    except Exception:
        data = _parse_lines(path)
    acc = data[:, 0:3] * ADXL_LSB2G          # 转 g
    sv = np.sqrt((acc ** 2).sum(axis=1))     # 合加速度 SV
    b, a = butter(4, [0.5, 10], btype="band", fs=FS)
    sv_f = filtfilt(b, a, sv)
    return sv, sv_f

rows = []
for sub in SUBJECTS:
    folder = os.path.join(ROOT, sub)
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        code = fname.split("_")[0]
        label = "跌倒" if code.startswith("F") else "日常"
        try:
            sv, sv_f = load_sv(os.path.join(folder, fname))
        except Exception as e:
            print("跳过", fname, e)
            continue
        rows.append({
            "file": fname, "subject": sub, "code": code, "label": label,
            "peak_raw_g": sv.max(),
            "peak_filt_g": sv_f.max(),
            "mean_g": sv_f.mean(),
        })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "sisfall_峰值统计_SA01-SA05.csv"),
          index=False, encoding="utf-8-sig")
print(f"共处理 {len(df)} 个文件：跌倒 {len(df[df.label=='跌倒'])} 个，"
      f"日常 {len(df[df.label=='日常'])} 个")

# ============ 图 1：三联对比图 ============
fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

# (a) 峰值直方图
ax = axes[0]
bins = np.linspace(0, max(df.peak_filt_g.max(), 1) * 1.05, 60)
ax.hist(df[df.label == "日常"].peak_filt_g, bins=bins, alpha=0.55,
        label="日常活动 (D01-D19)", color="#378ADD", density=True)
ax.hist(df[df.label == "跌倒"].peak_filt_g, bins=bins, alpha=0.55,
        label="跌倒 (F01-F15)", color="#E24B4A", density=True)
ax.set_xlabel("滤波后 SV 峰值 (g)")
ax.set_ylabel("概率密度")
ax.set_title("(a) 峰值分布：重叠区 = 阈值法失效区")
ax.legend()

# (b) 每类动作箱线图（按峰值中位数排序）
ax = axes[1]
order = df.groupby("code").peak_filt_g.median().sort_values().index
data_by_code = [df[df.code == c].peak_filt_g.values for c in order]
colors = ["#E24B4A" if c.startswith("F") else "#378ADD" for c in order]
bp = ax.boxplot(data_by_code, showfliers=True, flierprops=dict(markersize=2),
                patch_artist=True)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels(order, rotation=90, fontsize=8)
ax.set_ylabel("滤波后 SV 峰值 (g)")
ax.set_title("(b) 每类动作峰值（红=跌倒，蓝=日常）")

# (c) 阈值法性能曲线
ax = axes[2]
thr = np.linspace(0, df.peak_filt_g.max() * 1.02, 200)
is_fall = (df.label == "跌倒").values
recall = np.array([(df.peak_filt_g.values > t)[is_fall].mean() for t in thr])
false_alarm = np.array([(df.peak_filt_g.values > t)[~is_fall].mean() for t in thr])
f1 = []
for t in thr:
    pred = df.peak_filt_g.values > t
    tp = (pred & is_fall).sum()
    f1.append(2 * tp / (pred.sum() + is_fall.sum()) if (pred.sum() + is_fall.sum()) else 0)
f1 = np.array(f1)
best = f1.argmax()
ax.plot(thr, recall * 100, label="跌倒召回率", color="#E24B4A")
ax.plot(thr, (1 - false_alarm) * 100, label="日常正确率", color="#378ADD")
ax.plot(thr, f1 * 100, label="F1", color="#3B6D11")
ax.axvline(thr[best], color="#888780", ls="--", lw=1)
ax.annotate(f"最佳阈值≈{thr[best]:.2f} g\nF1={f1[best]*100:.1f}%",
            xy=(thr[best], f1[best] * 100), xytext=(thr[best] + 0.5, 60),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="#5F5E5A"))
ax.set_xlabel("判定阈值：SV峰值 > 阈值 → 报警 (g)")
ax.set_ylabel("%")
ax.set_title("(c) 单一阈值法的性能天花板")
ax.legend(loc="lower left")
ax.set_ylim(-5, 105)

fig.suptitle("SisFall 批量分析（SA01-SA05）：阈值法为什么挡不住慢速跌倒", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "sisfall_跌倒vs日常对比.png"), dpi=150)
print("图已保存")

# ============ 关键结论打印 ============
falls, adls = df[df.label == "跌倒"], df[df.label == "日常"]
print("\n--- 关键数字 ---")
print(f"跌倒峰值中位数 {falls.peak_filt_g.median():.2f} g，日常 {adls.peak_filt_g.median():.2f} g")
print(f"峰值最低的 10 个跌倒文件（最容易被漏报）：")
low10 = falls.nsmallest(10, "peak_filt_g")[["file", "code", "peak_filt_g"]]
print(low10.to_string(index=False))
print(f"峰值最高的 5 个日常文件（最容易被误报）：")
top5 = adls.nlargest(5, "peak_filt_g")[["file", "code", "peak_filt_g"]]
print(top5.to_string(index=False))
