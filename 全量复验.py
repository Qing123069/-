# -*- coding: utf-8 -*-
"""
全量复验（4505 条）—— 回应两个质疑
==========================================================
质疑 1：之前 99.5%/0.8% 只在 SA01-SA05 的 625 文件上算的，全数据集上呢？
质疑 2：阈值和规则在同一批数据上调参又报分，没有交叉验证。

本脚本：
  1. 在 SisFall 全部 38 名受试者、全部 txt 文件上复验规则
     【终态倾角 > 60° 且 >60° 持续 ≥ 3 秒】（与 04_姿态解算实验 同一方法）
  2. 输出逐类别（D01-D19 / F01-F15）混淆矩阵
  3. 受试者级 5 折交叉验证：参数在训练折上挑、在测试折上报，泛化口径
  4. SA / SE 分组报告（注意：老年组除 SE06 外没做跌倒动作）

数据事实（来自官方 Readme.txt + 原始论文 MDPI Sensors 2017）：
  - 设备（ADXL345±16g + ITG3200 + MMA8451Q±8g）只有一个，用腰带固定于腰部。
    ★ SisFall 没有腕部传感器 ★ 本实验全部为"腰部/躯干"数据。
  - 老年人（SE01-SE15）只做 ADL；例外 SE06（柔道教练）也做了跌倒。
  - 老年人免做 D06/D13/D18/D19；部分人有个人原因缺席其他动作。
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"E:\防摔倒检测\SisFall\SisFall_dataset\SisFall_dataset"
OUT = os.path.dirname(os.path.abspath(__file__))
FS = 200.0
ADXL_LSB2G = (2 * 16) / (2 ** 13)

ACT_NAMES = {
    "D01": "慢走", "D02": "快走", "D03": "慢跑", "D04": "快跑",
    "D05": "慢速上下楼梯", "D06": "快速上下楼梯",
    "D07": "缓慢坐半高椅-起身", "D08": "快速坐半高椅-起身",
    "D09": "缓慢坐矮椅-起身", "D10": "快速坐矮椅-起身",
    "D11": "起身失败瘫回椅子", "D12": "坐-缓慢躺-再坐起",
    "D13": "坐-快速躺-再坐起", "D14": "床上仰卧翻身",
    "D15": "屈膝缓慢下蹲-起立", "D16": "不屈膝缓慢弯腰-起立",
    "D17": "上下车保持坐姿", "D18": "行走中被绊(未倒)", "D19": "轻跳(未倒)",
    "F01": "行走滑倒-前倒", "F02": "行走滑倒-后倒", "F03": "行走滑倒-侧倒",
    "F04": "行走绊倒-前倒", "F05": "慢跑绊倒-前倒",
    "F06": "晕厥垂直倒地", "F07": "晕厥扶桌倒地",
    "F08": "起身时前倒", "F09": "起身时侧倒",
    "F10": "坐下时前倒", "F11": "坐下时后倒", "F12": "坐下时侧倒",
    "F13": "坐姿晕厥前倒", "F14": "坐姿晕厥后倒", "F15": "坐姿晕厥侧倒",
}

# ---------------------------------------------------------------- 特征计算
def load_acc(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read().replace(";", " ").replace(",", " ")
    arr = np.array(s.split(), dtype=float)
    if arr.size < 9:
        return None
    d = arr.reshape(-1, 9)
    d = d[np.isfinite(d).all(axis=1)]
    return d[:, 0:3] * ADXL_LSB2G

def tilt_accel(acc):
    b, a = butter(2, 0.5, btype="low", fs=FS)
    sm = np.stack([filtfilt(b, a, acc[:, k]) for k in range(3)], axis=1)
    n = np.linalg.norm(sm, axis=1, keepdims=True)
    n[n == 0] = 1e-9
    sm = sm / n
    ref = sm[:int(FS)].mean(axis=0)
    rn = np.linalg.norm(ref)
    if rn == 0:
        return None
    ref /= rn
    return np.degrees(np.arccos(np.clip(sm @ ref, -1, 1)))

def sv_filtered(acc):
    sv = np.sqrt((acc ** 2).sum(axis=1))
    b, a = butter(4, [0.5, 10], btype="band", fs=FS)
    return filtfilt(b, a, sv)

def features(path):
    acc = load_acc(path)
    if acc is None or len(acc) < 5 * int(FS):
        return None
    tilt = tilt_accel(acc)
    if tilt is None:
        return None
    sv_f = sv_filtered(acc)
    event = int(np.argmax(sv_f))
    s1 = min(len(tilt) - 1, event + 3 * int(FS))
    e1 = min(len(tilt), event + 8 * int(FS))
    if e1 - s1 < int(FS):
        s1, e1 = min(len(tilt) - 1, event + int(FS)), len(tilt)
    return {
        "终态倾角_中位": float(np.median(tilt[s1:e1])),
        "倾角超60度时长_s": float((tilt[event:] > 60).sum() / FS),
        "事件时刻_s": event / FS,
        "文件时长_s": len(tilt) / FS,
    }

def predict(feat, th, dur):
    return feat["终态倾角_中位"] > th and feat["倾角超60度时长_s"] >= dur

# ---------------------------------------------------------------- 遍历数据集
rows, skipped = [], []
n_done = 0
for sub in sorted(os.listdir(ROOT)):
    folder = os.path.join(ROOT, sub)
    if not os.path.isdir(folder):
        continue
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        code = fname.split("_")[0]
        try:
            feat = features(os.path.join(folder, fname))
        except Exception as e:
            feat = None
            print("解析失败:", fname, e)
        if feat is None:
            skipped.append(fname)
            continue
        rows.append({
            "文件": fname, "受试者": sub,
            "年龄组": "青年SA" if sub.startswith("SA") else "老年SE",
            "动作码": code,
            "动作": ACT_NAMES.get(code, code),
            "标签": "跌倒" if code.startswith("F") else "日常",
            **feat,
        })
        n_done += 1
        if n_done % 500 == 0:
            print(f"已处理 {n_done} 个文件", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "逐文件特征_全量.csv"), index=False, encoding="utf-8-sig")
print(f"有效文件 {len(df)} 个，跳过 {len(skipped)} 个")

# ---------------------------------------------------------------- 固定规则全量结果
TH, DUR = 60.0, 3.0
df["预测为跌倒"] = (df.终态倾角_中位 > TH) & (df.倾角超60度时长_s >= DUR)

falls = df[df.标签 == "跌倒"]
adls = df[df.标签 == "日常"]
tp = int(falls.预测为跌倒.sum()); fn = int(len(falls) - tp)
fp = int(adls.预测为跌倒.sum()); tn = int(len(adls) - fp)
prec = tp / (tp + fp) if tp + fp else 0
rec = tp / (tp + fn) if tp + fn else 0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
print(f"\n===== 固定规则[倾角>{TH:.0f}° 且持续≥{DUR:.0f}s] 全量结果 =====")
print(f"跌倒: 召回 {rec*100:.2f}% ({tp}/{tp+fn})")
print(f"日常: 误报 {fp}/{len(adls)} = {fp/len(adls)*100:.2f}%")
print(f"精确率 {prec*100:.2f}%  F1 {f1*100:.2f}%")

print("\n--- 按年龄组 ---")
for g, gd in df.groupby("年龄组"):
    gf = gd[gd.标签 == "跌倒"]; ga = gd[gd.标签 == "日常"]
    r = gf.预测为跌倒.mean() * 100 if len(gf) else float("nan")
    fa = ga.预测为跌倒.mean() * 100 if len(ga) else float("nan")
    print(f"  {g}: 跌倒召回 {r:.2f}% (n={len(gf)}) ｜ 日常误报 {fa:.2f}% (n={len(ga)})")

print("\n--- 逐类别混淆矩阵 ---")
cat = df.groupby(["标签", "动作码", "动作"]).agg(
    文件数=("文件", "count"),
    预测为跌倒数=("预测为跌倒", "sum"),
    终态倾角中位_50=("终态倾角_中位", "median"),
).reset_index()
cat["比例%"] = (cat.预测为跌倒数 / cat.文件数 * 100).round(2)
cat["召回/误报%"] = cat["比例%"]
cat = cat.sort_values(["标签", "动作码"])
cat.to_csv(os.path.join(OUT, "逐类别混淆矩阵_固定规则.csv"), index=False, encoding="utf-8-sig")
for _, r in cat.iterrows():
    kind = "召回" if r.标签 == "跌倒" else "误报"
    print(f"  [{r.动作码}] {r.动作:<14} n={r.文件数:>4}  触发={int(r.预测为跌倒数):>4}  {kind}={r['比例%']:>6.2f}%")

# ---------------------------------------------------------------- 受试者级 5 折交叉验证
print("\n===== 受试者级 5 折交叉验证 =====")
rng = np.random.RandomState(42)
sa_subs = sorted([s for s in df.受试者.unique() if s.startswith("SA")])
se_subs = sorted([s for s in df.受试者.unique() if s.startswith("SE")])
sa_perm = rng.permutation(sa_subs); se_perm = rng.permutation(se_subs)
folds = [[] for _ in range(5)]
for i, s in enumerate(sa_perm): folds[i % 5].append(s)
for i, s in enumerate(se_perm): folds[i % 5].append(s)

GRID = [(th, dur) for th in (50, 55, 60, 65, 70, 75) for dur in (1, 2, 3, 4, 5)]

def metrics(subset, th, dur):
    f = subset[subset.标签 == "跌倒"]; a = subset[subset.标签 == "日常"]
    tp = int(((f.终态倾角_中位 > th) & (f.倾角超60度时长_s >= dur)).sum()) if len(f) else 0
    fp = int(((a.终态倾角_中位 > th) & (a.倾角超60度时长_s >= dur)).sum()) if len(a) else 0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / len(f) if len(f) else 0.0
    f1v = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return rec, fp / len(a) if len(a) else 0.0, f1v

cv_rows = []
for k, members in enumerate(folds):
    test = df[df.受试者.isin(members)]
    train = df[~df.受试者.isin(members)]
    best = max(GRID, key=lambda p: metrics(train, *p)[2])
    tr_r, tr_fp, tr_f1 = metrics(train, *best)
    te_r, te_fp, te_f1 = metrics(test, *best)
    fx_r, fx_fp, fx_f1 = metrics(test, TH, DUR)
    cv_rows.append({"折": k + 1, "测试受试者": ",".join(members),
                    "选中参数": f"{best[0]}°/{best[1]}s",
                    "测试召回%": te_r * 100, "测试误报%": te_fp * 100, "测试F1%": te_f1 * 100,
                    "固定规则召回%": fx_r * 100, "固定规则误报%": fx_fp * 100, "固定规则F1%": fx_f1 * 100})
    print(f"  折{k+1}: 参数={best[0]}°/{best[1]}s  测试 召回{te_r*100:5.1f}% 误报{te_fp*100:4.1f}% F1 {te_f1*100:5.1f}%"
          f"  ｜ 固定60°/3s 召回{fx_r*100:5.1f}% 误报{fx_fp*100:4.1f}% F1 {fx_f1*100:5.1f}%")

cv = pd.DataFrame(cv_rows)
cv.to_csv(os.path.join(OUT, "交叉验证_受试者级5折.csv"), index=False, encoding="utf-8-sig")
print(f"\n  交叉验证汇总: 测试召回 {cv['测试召回%'].mean():.1f}±{cv['测试召回%'].std():.1f}%  "
      f"测试误报 {cv['测试误报%'].mean():.2f}±{cv['测试误报%'].std():.2f}%  测试F1 {cv['测试F1%'].mean():.1f}±{cv['测试F1%'].std():.1f}%")
print(f"  固定规则同折: 召回 {cv['固定规则召回%'].mean():.1f}±{cv['固定规则召回%'].std():.1f}%  "
      f"误报 {cv['固定规则误报%'].mean():.2f}±{cv['固定规则误报%'].std():.2f}%  F1 {cv['固定规则F1%'].mean():.1f}±{cv['固定规则F1%'].std():.1f}%")

# ---------------------------------------------------------------- 图：逐类别触发率
fig, ax = plt.subplots(figsize=(11, 12))
cc = cat.sort_values("比例%")
colors = ["#E24B4A" if t == "跌倒" else "#378ADD" for t in cc.标签]
y = np.arange(len(cc))
ax.barh(y, cc["比例%"], color=colors, alpha=0.75)
ax.set_yticks(y)
ax.set_yticklabels([f"{r.动作码} {r.动作}" for _, r in cc.iterrows()], fontsize=9)
ax.axvline(50, color="#888780", ls="--", lw=1)
ax.set_xlabel("被规则判为「跌倒」的文件比例（%）")
ax.set_title(f"全量 {len(df)} 条记录 · 规则[终态倾角>{TH:.0f}°且持续≥{DUR:.0f}s]\n红色=跌倒（应为100%）｜蓝色=日常（应接近0%）", fontsize=13)
for yi, (v, n) in enumerate(zip(cc["比例%"], cc.文件数)):
    ax.text(v + 1, yi, f"{v:.0f}% (n={n})", va="center", fontsize=8, color="#5F5E5A")
ax.set_xlim(0, 115)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "逐类别触发率_全量.png"), dpi=150)
print("\n图已保存: 逐类别触发率_全量.png")
