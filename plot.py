import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from settings import MAP

#load latest log file
files = glob.glob("logs/run_log_*.csv")
if not files:
    raise RuntimeError("NO LOG PAPI")
latest_file = max(files, key=os.path.getmtime)
print(f"Loading data from: {latest_file}")
df = pd.read_csv(latest_file)

ep=df.groupby("episode").agg(mean_lu=('local_uncertainty','mean'),
                             mean_fi=('frontier_ignorance','mean'),
                             mean_gc=('ghost_certainty','mean'),
                             mean_te=('tiles_explored','mean'),
                             mean_nt=('new_tiles','mean'),
                             mean_pe=('mean_prediction_error','mean'))

# Diagnostic calculations[Revisist Rate, New Tiles per Steps, Seen Tiles out of Total]
df["is_idle"] = df["action"] == "idle"
df["visited_before"] = df.duplicated(subset=["episode", "x", "y"])
revisit_rate = df.groupby("episode")["visited_before"].mean()
revisit_idle = df[df["is_idle"]].groupby("episode")["visited_before"].mean()
revisit_move = df[~df["is_idle"]].groupby("episode")["visited_before"].mean()
print("\\\DIAGNOSTIC//")
print("Revisit rate per episode:", revisit_rate.tail(10))
print("Revisit rate when idle per episode:", revisit_idle.tail(10))
print("Revisit rate when moving per episode:", revisit_move.tail(10))

new_tiles_per_1000 = df["new_tiles"].sum() / (len(df) / 1000)
print("New tiles per 1000 steps(global):", new_tiles_per_1000)

WINDOW = 1000
df["new_tiles_window"]=df["new_tiles"].rolling(window=WINDOW, min_periods=WINDOW).sum()
new_tiles_window = df["new_tiles_window"]
print(f"New tiles per {WINDOW} steps (rolling):", new_tiles_window.tail(10))

H = len(MAP)
W = len(MAP[0])
sum = df.iloc[-1]
total_tiles = H * W
seen_tiles = int(sum["tiles_explored"]*total_tiles)

print(f"There were {seen_tiles} tiles out of {total_tiles} explored in the last episode.")
print("-----------------------------------")
#Idleness through time
window = 500
df["idle_roll"]=df["is_idle"].rolling(window, min_periods=window).mean()
df["lu_roll"]=df["local_uncertainty"].rolling(window, min_periods=window).mean()
df["fi_roll"]=df["frontier_ignorance"].rolling(window, min_periods=window).mean()
df["pe_roll"]=df["mean_prediction_error"].rolling(window,min_periods=window).mean()

# Conditional idle probability
df["local_uncertainty"] = pd.to_numeric(df["local_uncertainty"], errors="coerce")
df["frontier_ignorance"] = pd.to_numeric(df["frontier_ignorance"], errors="coerce")
df["mean_prediction_error"] = pd.to_numeric(df["mean_prediction_error"], errors="coerce")
df["is_idle"] = (df["action"] == "idle")

# Drop rows where LU/FI missing (shouldn't happen often, but makes masks safe)
df2 = df.dropna(subset=["local_uncertainty", "frontier_ignorance","mean_prediction_error"]).copy()

lu_thresh = df2["local_uncertainty"].quantile(0.10)
fi_thresh = df2["frontier_ignorance"].quantile(0.10)
pe_thresh = df2["mean_prediction_error"].quantile(0.10)

low_mask = ((df2["local_uncertainty"] <= lu_thresh) & (df2["frontier_ignorance"] <= fi_thresh) & (df2["mean_prediction_error"]<=pe_thresh)).to_numpy()

p_idle_low = df2["is_idle"].to_numpy()[low_mask].mean() if low_mask.any() else float("nan")
p_idle_high = df2["is_idle"].to_numpy()[~low_mask].mean() if (~low_mask).any() else float("nan")

print("Idle vs Epistemic State")
print("LU threshold (10%):", lu_thresh)
print("FI threshold (10%):", fi_thresh)
print("MPE treshold (10%);", pe_thresh)
print("P(idle | low LU & low FI):", p_idle_low)
print("P(idle | otherwise):      ", p_idle_high)

# Periods of Idleness
df["new_tiles_1000"] = df["new_tiles"].rolling(WINDOW, min_periods=WINDOW).sum()
df["lu_roll"] = df["local_uncertainty"].rolling(WINDOW, min_periods=WINDOW).mean()
df["fi_roll"] = df["frontier_ignorance"].rolling(WINDOW, min_periods=WINDOW).mean()
df["pe_roll"] = df["mean_prediction_error"].rolling(WINDOW, min_periods=WINDOW).mean()

# Less brittle thresholds: use 25% instead of 10% to avoid near-zero collapse
lu_thresh_idle = df["local_uncertainty"].quantile(0.25)
fi_thresh_idle = df["frontier_ignorance"].quantile(0.25)
pe_thresh_idle = df["mean_prediction_error"].quantile(0.25)

df["epistemic_idleness"] = (
    (df["new_tiles_1000"] <= 0) &
    (df["lu_roll"] <= lu_thresh_idle) &
    (df["fi_roll"] <= fi_thresh_idle) &
    (df["pe_roll"] <= pe_thresh_idle)
)

# contiguous blocks
df["idle_block2"] = (df["epistemic_idleness"] != df["epistemic_idleness"].shift()).cumsum()

idle_segments = (
    df[df["epistemic_idleness"]]
    .groupby("idle_block2")
    .agg(
        episode=("episode", "min"),
        start_step=("step_in_episode", "min"),
        end_step=("step_in_episode", "max"),
        duration_steps=("epistemic_idleness", "size"),
        mean_pe=("mean_prediction_error", "mean"),
        mean_lu=("local_uncertainty", "mean"),
        mean_fi=("frontier_ignorance", "mean"),
        new_tiles_sum=("new_tiles", "sum"),
        idle_frac=("is_idle", "mean"),
    )
    .sort_values("duration_steps", ascending=False)
)

print("Longest Idleness Segments:", idle_segments.head(5))

q = df[df["epistemic_idleness"]].copy()
action_dist_q = q["action"].value_counts(normalize=True)

nq = df[~df["epistemic_idleness"]].copy()
action_dist_nq = nq["action"].value_counts(normalize=True)

print("\nAction distribution DURING epistemic idleness:")
print(action_dist_q)
print("\nAction distribution OUTSIDE epistemic idleness:")
print(action_dist_nq)
print("-----------------------------------")

ts = os.path.basename(latest_file).replace("run_log_", "").replace(".csv", "")

print(df.head())

# --- Plot 1: Local uncertainty over time ---
plt.figure()
plt.plot(ep.index, ep["mean_lu"])
plt.xlabel("Episode")
plt.ylabel("Mean Local Uncertainty")
plt.title("Mean Local Uncertainty Over Episodes")
plt.savefig(f"logs/mean_local_uncertainty_{ts}.png")
plt.close()

# --- Plot 2: Frontier ignorance over time ---
plt.figure()
plt.plot(ep.index, ep["mean_fi"])
plt.xlabel("Episode")
plt.ylabel("Mean Frontier Ignorance")
plt.title("Mean Frontier Ignorance Over Episodes")
plt.savefig(f"logs/mean_frontier_ignorance_{ts}.png")
plt.close()


# --- Plot 3: Action distribution ---
plt.figure()
df["action"].value_counts().plot(kind="bar")
plt.xlabel("Action")
plt.ylabel("Count")
plt.title("Action Distribution")
plt.savefig(f"logs/action_distribution_{ts}.png")
plt.close()

# --- Plot 5: Ghost certainty over time ---
plt.figure()
plt.plot(ep.index, ep["mean_gc"])
plt.xlabel("Episode")
plt.ylabel("Mean Ghost Certainty")
plt.title("Mean Ghost Certainty Over Episodes")
plt.savefig(f"logs/mean_ghost_certainty_{ts}.png")
plt.close()

# --- Plot 6: Tiles explored over time ---
plt.figure()
plt.plot(ep.index, ep["mean_te"])
plt.xlabel("Episode")
plt.ylabel("Mean Tiles Explored")
plt.title("Mean Tiles Explored Over Episodes")
plt.savefig(f"logs/mean_tiles_explored_{ts}.png")
plt.close()

#--- Plot 7: New tiles discovered over time ---
plt.figure()
plt.plot(ep.index, ep["mean_nt"])
plt.xlabel("Episode")
plt.ylabel("Mean New Tiles Discovered")
plt.title("Mean New Tiles Discovered Over Episodes")
plt.savefig(f"logs/mean_new_tiles_{ts}.png")
plt.close()

#---Plot 8: Mean Prediction Error over time ---
plt.figure()
plt.plot(ep.index, ep["mean_pe"])
plt.xlabel("Episode")
plt.ylabel("Mean Prediction Error")
plt.title("Mean Prediction Error over Episodes")
plt.savefig(f"logs/mean_prediction_error_{ts}.png")
plt.close

print("Plots saved to logs/ with timestamp:", ts)


# ------------- CORRELATION ------------------

needed = ["mean_prediction_error", "local_uncertainty", "frontier_ignorance", "episode", "action"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in CSV: {missing}")

df["mean_prediction_error"]= pd.to_numeric(df["mean_prediction_error"], errors="coerce")
df["local_uncertainty"]=pd.to_numeric(df["local_uncertainty"], errors="coerce")
df["frontier_ignorance"]=pd.to_numeric(df["frontier_ignorance"], errors="coerce")

#behaviour
df["is_idle"]=(df["action"]=="idle")
df["is_move"]=(~df["is_idle"]).astype(int)

df2 = df.dropna(subset=["mean_prediction_error", "local_uncertainty", "frontier_ignorance"]).copy()

drop_all_zero= True
if drop_all_zero:
    mask_zero= (
        (df2["mean_prediction_error"]==0) &
        (df2["local_uncertainty"]==0) &
        (df2["frontier_ignorance"]==0)
    )
    df2 = df2.loc[~mask_zero].copy()

print("------------ SPEARMAN CORRELATION (GLOBAL) --------------")

corr_cols = ["mean_prediction_error", "local_uncertainty", "frontier_ignorance", "is_move"]
global_corr= df2[corr_cols].corr(method="spearman")
global_corr.to_csv(f"logs/global_spearman_{ts}.csv")
print(global_corr)

print("------------ SPEARMAN CORRELATION (PER EPISODE) --------------")
def spearman_pair(g, a, b): #g is a dataframe
    if len(g)<10:
        return np.nan
    if g[a].nunique()<2 or g[b].nunique()<2:
        return np.nan
    return g[[a,b]].corr(method="spearman").iloc[0,1]

rows=[]
for ep, g in df2.groupby("episode"):
    rows.append({
        "episode":ep,
        "rho_PE_LU": spearman_pair(g,"mean_prediction_error","local_uncertainty"),
        "rho_PE_FI": spearman_pair(g,"mean_prediction_error","frontier_ignorance"),
        "rho_LU_FI": spearman_pair(g, "local_uncertainty", "frontier_ignorance"),
        "rho_PE_move": spearman_pair(g,"mean_prediction_error", "is_move"),
        "rho_LU_move": spearman_pair(g,"local_uncertainty", "is_move"),
        "rho_FI_move": spearman_pair(g, "frontier_ignorance", "is_move"),
        "n_steps":len(g)
    })

per_ep_corr= pd.DataFrame(rows).sort_values("episode")
print(per_ep_corr.tail(15))

per_ep_corr.to_csv(f"logs/per_episode_spearman_correlations{ts}.csv", index=False)
print("\nSaved: per_episode_spearman_correlations.csv")

print("------------ SPEARMAN CORRELATION (PHASES) --------------")

max_ep = int(df2["episode"].max())
cut1 = max(1, max_ep//3)
cut2= max(2, 2*max_ep//3)

def phase_label(ep):
    if ep <= cut1:
        return "early"
    elif ep <= cut2:
        return "mid"
    else:
        return "late"
df2["phase"]=df2["episode"].apply(phase_label)

phase_rows = []
for ph, g in df2.groupby("phase"):
    phase_corr= g[corr_cols].corr(method="spearman")
    phase_rows.append({
        "phase":ph,
        "n_steps": len(g),
        "rho_PE_LU": phase_corr.loc["mean_prediction_error", "local_uncertainty"],
        "rho_PE_FI": phase_corr.loc["mean_prediction_error", "frontier_ignorance"],
        "rho_LU_FI": phase_corr.loc["local_uncertainty", "frontier_ignorance"],
        "rho_PE_move": phase_corr.loc["mean_prediction_error", "is_move"]
    })

phase_corr_df = pd.DataFrame(phase_rows).sort_values("phase")
print(phase_corr_df)


# ------------- PE VARIANCE v Movement --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 0) Basic flags ---
df["is_idle"] = (df["action"] == "idle")
df["is_move"] = ~df["is_idle"]

pe_col = "mean_prediction_error"   # your per-step PE column name

# Safety: drop rows where PE is missing
df_pe = df.dropna(subset=[pe_col]).copy()

# --- 1) Global variance / std (move vs idle) ---
global_stats = (
    df_pe.groupby("is_move")[pe_col]
    .agg(n="size", mean="mean", var="var", std="std", median="median")
)
global_stats.index = global_stats.index.map({True: "move", False: "idle"})
global_stats.to_csv(f"logs/pe_variance_global_move_vs_idle_{ts}.csv")
print("\nSaved: pe_variance_global_move_vs_idle.csv")

# --- 2) Per-episode variance (move vs idle) ---
ep_stats = (
    df_pe.groupby(["episode", "is_move"])[pe_col]
    .agg(n="size", mean="mean", var="var", std="std", median="median")
    .reset_index()
)
ep_stats["state"] = ep_stats["is_move"].map({True: "move", False: "idle"})
ep_stats.to_csv(f"logs/pe_variance_per_episode_move_vs_idle_{ts}.csv", index=False)
print("Saved: pe_variance_per_episode_move_vs_idle.csv")

# Pivot for plotting
ep_var = ep_stats.pivot(index="episode", columns="state", values="var")
ep_std = ep_stats.pivot(index="episode", columns="state", values="std")

# --- 3) Plot: per-episode PE variance for move vs idle ---
plt.figure()
ep_var.plot(kind="line")
plt.title("Per-episode PE variance: move vs idle")
plt.xlabel("Episode")
plt.ylabel("Var(PE)")
plt.tight_layout()
plt.savefig(f"logs/pe_variance_per_episode_move_vs_idle_{ts}.png", dpi=200)
plt.close()
print("Saved: pe_variance_per_episode_move_vs_idle.png")

# --- 4) Rolling variance (captures bursts/spikes) ---
# Choose a window that matches your intuition about "short-term dynamics"
ROLL_W = 200
df_pe["pe_var_roll"] = df_pe[pe_col].rolling(ROLL_W, min_periods=ROLL_W).var()

# Compare rolling variance distributions
roll_move = df_pe.loc[df_pe["is_move"], "pe_var_roll"].dropna()
roll_idle = df_pe.loc[df_pe["is_idle"], "pe_var_roll"].dropna()

roll_summary = pd.DataFrame({
    "group": ["move", "idle"],
    "n": [len(roll_move), len(roll_idle)],
    "mean": [roll_move.mean(), roll_idle.mean()],
    "median": [roll_move.median(), roll_idle.median()],
    "var": [roll_move.var(), roll_idle.var()],
    "std": [roll_move.std(), roll_idle.std()],
})
roll_summary.to_csv(f"logs/pe_rolling_variance_summary_move_vs_idle_{ts}.csv", index=False)
print("Saved: pe_rolling_variance_summary_move_vs_idle.csv")

# --- 5) Plot: rolling variance (boxplot) ---
plt.figure()
plt.boxplot([roll_move.values, roll_idle.values], labels=["move", "idle"], showfliers=False)
plt.title(f"Rolling PE variance (window={ROLL_W})")
plt.ylabel("Rolling Var(PE)")
plt.tight_layout()
plt.savefig(f"logs/pe_rolling_variance_boxplot_move_vs_idle_{ts}.png", dpi=200)
plt.close()
print("Saved: pe_rolling_variance_boxplot_move_vs_idle.png")
