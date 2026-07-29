import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from datetime import datetime
from settings import MAP

# -------------------------------------------------------
# Config
# -------------------------------------------------------
CONDITIONS = ["decay_on", "decay_off", "c1_random", "c5_pure_pe"]
CONDITION_COLORS = {
    "decay_on":   "#1f77b4",  # blue
    "decay_off":  "#ff7f0e",  # orange
    "c1_random":  "#2ca02c",  # green
    "c5_pure_pe": "#d62728",  # red
}
SEEDS = ["1", "2", "3"]
H = len(MAP)
W = len(MAP[0])
total_tiles = H * W
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
WINDOW = 1000
ROLL_W = 200
pe_col = "mean_prediction_error"

# -------------------------------------------------------
# Load all logs
# -------------------------------------------------------
condition_files = {c: sorted(glob.glob(f"logs/run_log_{c}_seed*.csv")) for c in CONDITIONS}
all_files = [f for files in condition_files.values() for f in files]
if not all_files:
    raise RuntimeError("NO LOG FILES FOUND. Run conditions.py first.")

dfs = {}
for condition, files in condition_files.items():
    for f in files:
        try:
            seed = f.split("seed")[1].split("_")[0]
        except IndexError:
            seed = "?"
        label = f"{condition}_s{seed}"
        dfs[label] = pd.read_csv(f)
        dfs[label]["condition"] = condition
        dfs[label]["seed"] = seed

print(f"Loaded {len(dfs)} log files: {list(dfs.keys())}")

# -------------------------------------------------------
# Helper: aggregate per episode
# -------------------------------------------------------
def agg_episodes(df):
    return df.groupby("episode").agg(
        mean_lu=("local_uncertainty", "mean"),
        mean_fi=("frontier_ignorance", "mean"),
        mean_gc=("ghost_certainty", "mean"),
        mean_te=("tiles_explored", "mean"),
        mean_nt=("new_tiles", "mean"),
        mean_pe=(pe_col, "mean"),
    )

# -------------------------------------------------------
# Helper: extract condition from label safely
# -------------------------------------------------------
def get_condition(label):
    for c in CONDITIONS:
        if label.startswith(c + "_s"):
            return c
    return "unknown"

# -------------------------------------------------------
# Legend handles (shared across all figures)
# -------------------------------------------------------
legend_handles = [
    mlines.Line2D([], [], color=CONDITION_COLORS[c], linewidth=2.5, label=c)
    for c in CONDITIONS
]

# -------------------------------------------------------
# DIAGNOSTICS
# -------------------------------------------------------
print("\n========== DIAGNOSTICS ==========")
for condition in CONDITIONS:
    cdfs = {k: v for k, v in dfs.items() if get_condition(k) == condition}
    if not cdfs:
        print(f"\n[{condition}] No data found.")
        continue
    print(f"\n--- {condition} ---")
    for label, df in cdfs.items():
        df["is_idle"] = df["action"] == "idle"
        df["visited_before"] = df.duplicated(subset=["episode", "x", "y"])
        revisit_rate = df.groupby("episode")["visited_before"].mean()
        new_tiles_per_1000 = df["new_tiles"].sum() / (len(df) / 1000)
        seen_tiles = int(df.iloc[-1]["tiles_explored"] * total_tiles)
        print(f"  [{label}] Revisit rate (last 5 ep): {revisit_rate.tail(5).values.round(3)}")
        print(f"  [{label}] New tiles per 1000 steps: {new_tiles_per_1000:.2f}")
        print(f"  [{label}] Tiles explored (last step): {seen_tiles}/{total_tiles}")

# -------------------------------------------------------
# FIGURE SET 1: Epistemic metrics — one figure per seed
# -------------------------------------------------------
metric_config = {
    "mean_lu": ("Mean Local Uncertainty", "LU"),
    "mean_fi": ("Mean Frontier Ignorance", "FI"),
    "mean_pe": ("Mean Prediction Error", "PE"),
    "mean_gc": ("Mean Ghost Certainty", "GC"),
    "mean_te": ("Mean Tiles Explored", "Coverage"),
    "mean_nt": ("Mean New Tiles per Step", "New Tiles"),
}

for seed in SEEDS:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"CuriousMan — Epistemic Metrics (Seed {seed})",
                 fontsize=14, fontweight="bold")

    ax_map = {
        "mean_lu": axes[0, 0], "mean_fi": axes[0, 1], "mean_pe": axes[0, 2],
        "mean_gc": axes[1, 0], "mean_te": axes[1, 1], "mean_nt": axes[1, 2],
    }

    seed_labels = {label: df for label, df in dfs.items()
                   if label.endswith(f"_s{seed}")}

    for label, df in seed_labels.items():
        condition = get_condition(label)
        color = CONDITION_COLORS.get(condition, "gray")
        ep = agg_episodes(df)

        for metric, (title, ylabel) in metric_config.items():
            ax = ax_map[metric]
            ax.plot(ep.index, ep[metric], color=color, linewidth=2,
                    label=condition)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)

    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fname = f"logs/panel_epistemic_seed{seed}_{ts}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

# -------------------------------------------------------
# FIGURE SET 2: Behaviour & PE Variance — one figure per seed
# -------------------------------------------------------
for seed in SEEDS:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"CuriousMan — Behaviour & PE Variance (Seed {seed})",
                 fontsize=14, fontweight="bold")

    seed_labels = {label: df for label, df in dfs.items()
                   if label.endswith(f"_s{seed}")}

    # --- Panel 1: Action distribution ---
    ax_act = axes[0]
    action_counts = {}
    for label, df in seed_labels.items():
        condition = get_condition(label)
        action_counts[condition] = df["action"].value_counts(normalize=True)

    actions = ["left", "right", "up", "down", "idle"]
    x = np.arange(len(actions))
    width = 0.2
    for i, condition in enumerate(CONDITIONS):
        if condition not in action_counts:
            continue
        vals = [action_counts[condition].get(a, 0) for a in actions]
        ax_act.bar(x + i * width, vals, width, label=condition,
                   color=CONDITION_COLORS[condition], alpha=0.85)
    ax_act.set_xticks(x + width * 1.5)
    ax_act.set_xticklabels(actions, fontsize=9)
    ax_act.set_title(f"Action Distribution (Seed {seed})")
    ax_act.set_ylabel("Proportion")
    ax_act.legend(fontsize=8)

    # --- Panel 2: PE Variance move vs idle ---
    ax_pev = axes[1]
    box_data_move, box_data_idle, box_labels = [], [], []

    for condition in CONDITIONS:
        label = f"{condition}_s{seed}"
        if label not in seed_labels:
            continue
        df = seed_labels[label].dropna(subset=[pe_col]).copy()
        df["is_move"] = df["action"] != "idle"
        box_data_move.append(df.loc[df["is_move"], pe_col].values)
        box_data_idle.append(df.loc[~df["is_move"], pe_col].values)
        box_labels.append(condition.replace("_", "\n"))

    positions_move = np.arange(1, len(box_labels) * 3, 3)
    positions_idle = positions_move + 1

    ax_pev.boxplot(box_data_move, positions=positions_move, widths=0.7,
                   showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#aec7e8", alpha=0.8))
    ax_pev.boxplot(box_data_idle, positions=positions_idle, widths=0.7,
                   showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#ffbb78", alpha=0.8))
    ax_pev.set_xticks(positions_move + 0.5)
    ax_pev.set_xticklabels(box_labels, fontsize=8)
    ax_pev.set_title(f"PE Distribution: Move vs Idle (Seed {seed})")
    ax_pev.set_ylabel("Mean Prediction Error")
    move_patch = mlines.Line2D([], [], color="#aec7e8", linewidth=6, label="move")
    idle_patch = mlines.Line2D([], [], color="#ffbb78", linewidth=6, label="idle")
    ax_pev.legend(handles=[move_patch, idle_patch], fontsize=8)

    # --- Panel 3: Rolling PE variance ---
    ax_roll = axes[2]
    for label, df in seed_labels.items():
        condition = get_condition(label)
        color = CONDITION_COLORS.get(condition, "gray")
        df = df.dropna(subset=[pe_col]).copy()
        df["pe_var_roll"] = df[pe_col].rolling(ROLL_W, min_periods=ROLL_W).var()
        ax_roll.plot(df.index, df["pe_var_roll"], color=color,
                     linewidth=1.5, alpha=0.85, label=condition)
    ax_roll.set_title(f"Rolling PE Variance (window={ROLL_W}, Seed {seed})")
    ax_roll.set_xlabel("Step")
    ax_roll.set_ylabel("Var(PE)")
    ax_roll.legend(handles=legend_handles, fontsize=8)

    plt.tight_layout()
    fname = f"logs/panel_behaviour_seed{seed}_{ts}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

# -------------------------------------------------------
# SPEARMAN CORRELATIONS (per condition)
# -------------------------------------------------------
print("\n========== SPEARMAN CORRELATIONS ==========")

def spearman_pair(g, a, b):
    if len(g) < 10:
        return np.nan
    if g[a].nunique() < 2 or g[b].nunique() < 2:
        return np.nan
    return g[[a, b]].corr(method="spearman").iloc[0, 1]

corr_cols = ["mean_prediction_error", "local_uncertainty", "frontier_ignorance", "is_move"]

for condition in CONDITIONS:
    cdfs = [df for k, df in dfs.items() if get_condition(k) == condition]
    if not cdfs:
        continue
    combined = pd.concat(cdfs).copy()
    for col in ["mean_prediction_error", "local_uncertainty", "frontier_ignorance"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["is_idle"] = combined["action"] == "idle"
    combined["is_move"] = (~combined["is_idle"]).astype(int)

    df2 = combined.dropna(subset=["mean_prediction_error", "local_uncertainty",
                                   "frontier_ignorance"]).copy()
    mask_zero = (
        (df2["mean_prediction_error"] == 0) &
        (df2["local_uncertainty"] == 0) &
        (df2["frontier_ignorance"] == 0)
    )
    df2 = df2.loc[~mask_zero].copy()

    print(f"\n--- {condition} (global Spearman) ---")
    global_corr = df2[corr_cols].corr(method="spearman")
    print(global_corr.round(3))
    global_corr.to_csv(f"logs/global_spearman_{condition}_{ts}.csv")

    rows = []
    for ep_n, g in df2.groupby("episode"):
        rows.append({
            "episode": ep_n,
            "rho_PE_LU":   spearman_pair(g, "mean_prediction_error", "local_uncertainty"),
            "rho_PE_FI":   spearman_pair(g, "mean_prediction_error", "frontier_ignorance"),
            "rho_LU_FI":   spearman_pair(g, "local_uncertainty", "frontier_ignorance"),
            "rho_PE_move": spearman_pair(g, "mean_prediction_error", "is_move"),
            "rho_LU_move": spearman_pair(g, "local_uncertainty", "is_move"),
            "rho_FI_move": spearman_pair(g, "frontier_ignorance", "is_move"),
            "n_steps": len(g),
        })
    per_ep = pd.DataFrame(rows).sort_values("episode")
    per_ep.to_csv(f"logs/per_episode_spearman_{condition}_{ts}.csv", index=False)

    max_ep = int(df2["episode"].max())
    cut1 = max(1, max_ep // 3)
    cut2 = max(2, 2 * max_ep // 3)
    df2["phase"] = df2["episode"].apply(
        lambda e: "early" if e <= cut1 else ("mid" if e <= cut2 else "late")
    )
    print(f"  Phase correlations:")
    for ph, g in df2.groupby("phase"):
        pc = g[corr_cols].corr(method="spearman")
        print(f"    [{ph}] PE↔LU={pc.loc['mean_prediction_error','local_uncertainty']:.3f}  "
              f"PE↔FI={pc.loc['mean_prediction_error','frontier_ignorance']:.3f}  "
              f"PE↔move={pc.loc['mean_prediction_error','is_move']:.3f}")

# -------------------------------------------------------
# PE VARIANCE: Move vs Idle
# -------------------------------------------------------
print("\n========== PE VARIANCE: MOVE vs IDLE ==========")
for condition in CONDITIONS:
    cdfs = [df for k, df in dfs.items() if get_condition(k) == condition]
    if not cdfs:
        continue
    combined = pd.concat(cdfs).dropna(subset=[pe_col]).copy()
    combined["is_move"] = combined["action"] != "idle"
    stats = (
        combined.groupby("is_move")[pe_col]
        .agg(n="size", mean="mean", var="var", std="std", median="median")
    )
    stats.index = stats.index.map({True: "move", False: "idle"})
    print(f"\n--- {condition} ---")
    print(stats.round(8))
    stats.to_csv(f"logs/pe_variance_global_{condition}_{ts}.csv")

# -------------------------------------------------------
# FIGURE SET 3: Spearman correlations per episode — one figure per seed
# -------------------------------------------------------
spearman_metrics = [
    ("rho_PE_LU", "PE ↔ LU", "#1f77b4"),
    ("rho_PE_FI", "PE ↔ FI", "#ff7f0e"),
    ("rho_LU_FI", "LU ↔ FI", "#2ca02c"),
    ("rho_PE_move", "PE ↔ Move", "#d62728"),
]
 
for seed in SEEDS:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"CuriousMan — Spearman Correlations Per Episode (Seed {seed})",
                 fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()
 
    for ax_i, (metric, label, color) in enumerate(spearman_metrics):
        ax = axes_flat[ax_i]
        for condition in CONDITIONS:
            lbl = f"{condition}_s{seed}"
            if lbl not in dfs:
                continue
            # load per-episode spearman for this condition+seed
            # we need to recompute per-seed spearman since we only saved combined
            df = dfs[lbl].copy()
            for col in ["mean_prediction_error", "local_uncertainty",
                        "frontier_ignorance"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["is_idle"] = df["action"] == "idle"
            df["is_move"] = (~df["is_idle"]).astype(int)
            df2 = df.dropna(subset=["mean_prediction_error",
                                     "local_uncertainty",
                                     "frontier_ignorance"]).copy()
            mask_zero = (
                (df2["mean_prediction_error"] == 0) &
                (df2["local_uncertainty"] == 0) &
                (df2["frontier_ignorance"] == 0)
            )
            df2 = df2.loc[~mask_zero].copy()
 
            col_map = {
                "rho_PE_LU":   ("mean_prediction_error", "local_uncertainty"),
                "rho_PE_FI":   ("mean_prediction_error", "frontier_ignorance"),
                "rho_LU_FI":   ("local_uncertainty", "frontier_ignorance"),
                "rho_PE_move": ("mean_prediction_error", "is_move"),
            }
            a, b = col_map[metric]
 
            rows = []
            for ep_n, g in df2.groupby("episode"):
                rows.append({
                    "episode": ep_n,
                    "rho": spearman_pair(g, a, b)
                })
            per_ep = pd.DataFrame(rows).dropna()
 
            cond_color = CONDITION_COLORS.get(condition, "gray")
            ax.plot(per_ep["episode"], per_ep["rho"],
                    color=cond_color, linewidth=2,
                    label=condition)
 
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Spearman ρ")
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=8)
 
    plt.tight_layout()
    fname = f"logs/panel_spearman_seed{seed}_{ts}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
 
# -------------------------------------------------------
# FIGURE SET 4: Global Spearman heatmaps — one per condition
# -------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("CuriousMan — Global Spearman Correlation Matrices",
             fontsize=13, fontweight="bold")
 
short_labels = ["PE", "LU", "FI", "Move"]
 
for ax_i, condition in enumerate(CONDITIONS):
    ax = axes[ax_i]
    cdfs = [df for k, df in dfs.items() if get_condition(k) == condition]
    if not cdfs:
        ax.set_visible(False)
        continue
    combined = pd.concat(cdfs).copy()
    for col in ["mean_prediction_error", "local_uncertainty", "frontier_ignorance"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["is_idle"] = combined["action"] == "idle"
    combined["is_move"] = (~combined["is_idle"]).astype(int)
    df2 = combined.dropna(subset=["mean_prediction_error",
                                   "local_uncertainty",
                                   "frontier_ignorance"]).copy()
    mask_zero = (
        (df2["mean_prediction_error"] == 0) &
        (df2["local_uncertainty"] == 0) &
        (df2["frontier_ignorance"] == 0)
    )
    df2 = df2.loc[~mask_zero].copy()
    corr = df2[corr_cols].corr(method="spearman").values
 
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_title(condition, fontsize=10, fontweight="bold")
 
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr[i,j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(corr[i,j]) > 0.6 else "black")
 
plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="Spearman ρ")
plt.tight_layout()
fname = f"logs/panel_spearman_heatmaps_{ts}.png"
fig.savefig(fname, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {fname}")

print(f"\nAll plots and CSVs saved to logs/ with timestamp: {ts}")