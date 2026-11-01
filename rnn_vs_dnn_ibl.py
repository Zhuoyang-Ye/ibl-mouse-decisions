"""
DNN vs RNN on real mouse decision-making data (IBL)
=====================================================

WHAT THIS SCRIPT IS FOR
------------------------
Stage 2 of a two-stage pipeline (stage 1 is `fetch_ibl_data.py`, which caches
`data/ibl_trials.csv` from the public International Brain Laboratory
database). This script builds THREE small classifiers that all try to
predict a mouse's left/right choice on each trial, and compares them:

    1. DNN-narrow  : a feedforward MLP that only sees the CURRENT trial.
    2. DNN-window  : the same MLP, but also given a hand-picked, FIXED
                      window of the last K trials' choices/outcomes as
                      extra input features.
    3. RNN (GRU)   : a recurrent network that processes each session as a
                      sequence, carrying a hidden state from trial to
                      trial -- no window size to choose, no upper limit on
                      how far back it can (in principle) remember.

THE TASK, AND WHY IT'S A GOOD DNN vs RNN TEST CASE
----------------------------------------------------
On each trial, a mouse sees a visual grating on the LEFT or RIGHT at some
contrast (0 = invisible, 1 = maximum) and turns a wheel to report which
side it saw. The twist: `probabilityLeft`, the prior chance the stimulus
appears on the left, is silently held at 0.2 or 0.8 for blocks of ~50-100
trials, then flips -- with NO cue to the mouse. Well-trained mice learn to
track the current block from the recent history of stimuli/choices/reward,
and lean their choices that way -- especially on 0%-contrast trials, where
the image itself carries zero information and history is *all* the mouse
has to go on.

That's the crux of this demo: a feedforward network fed only the current
trial's stimulus is mathematically INCAPABLE of using block history --
there is no input for it to use. It will settle on one fixed prediction for
every 0%-contrast trial, no matter the block. An RNN, by contrast, can
carry the recent trial history in its hidden state and adjust its
prediction as the (hidden) block changes -- the same kind of computation
the mouse's brain has to be doing. We'll see this literally: on 0%-contrast
trials, the DNN-narrow's predicted probability is a flat line; the RNN's
isn't.

`DNN-window` sits in between: hand-engineering K lag features lets a plain
MLP fake having "memory" -- showing that DNN vs RNN is a difference of
*mechanism* and *convenience*, not of raw capability. But you have to
guess K correctly, and it's a fixed drop-off, not a smooth adaptive memory.

Run this file with (needs torch + pandas + matplotlib; no network needed,
no ibllib/one-api needed -- everything comes from the cached CSV):
    conda run -n code4fun python rnn_vs_dnn_ibl.py
It prints a running commentary and saves `rnn_vs_dnn_ibl.png`.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "ibl_trials.csv"
FIG_PATH = HERE / "rnn_vs_dnn_ibl.png"

LAG_K = 5              # window size for the "DNN-window" hand-engineered-memory model
TRAIN_FRACTION = 0.8   # chronological: earliest sessions train, latest sessions test
SWITCH_WINDOW = 25     # trials-since-block-switch to track in the adaptation plot

# =============================================================================
# 1. LOAD DATA + FEATURE ENGINEERING
# =============================================================================

def add_features(df: pd.DataFrame, lag_k: int = LAG_K) -> pd.DataFrame:
    """Trial-level feature engineering.

    `fetch_ibl_data.py` already dropped no-go trials and kept only the columns
    we need (plus three timing columns the models ignore). Here we turn the two mutually-exclusive contrast columns into one
    signed value, turn choice/feedback into 0/1 targets, and build "previous
    trial" features *within* each session (a session boundary is NOT a trial
    boundary -- trial 1 of a new session must not see trial 900 of
    yesterday's session as "the previous trial").
    """
    df = df.copy()
    # Positive = stimulus was on the RIGHT (at that contrast); negative = LEFT.
    # Exactly one of contrastLeft/contrastRight is non-NaN per trial.
    df["signed_contrast"] = df["contrastRight"].fillna(0) - df["contrastLeft"].fillna(0)
    df["choice_left"] = (df["choice"] == 1).astype(np.float32)   # 1 = chose left, 0 = chose right
    df["correct"] = (df["feedbackType"] == 1).astype(np.float32)

    g = df.groupby("session_id", sort=False)
    df["prev_choice_left"] = g["choice_left"].shift(1).fillna(0.0)
    df["prev_correct"] = g["correct"].shift(1).fillna(0.0)
    for lag in range(1, lag_k + 1):
        df[f"choice_left_lag{lag}"] = g["choice_left"].shift(lag).fillna(0.0)
        df[f"correct_lag{lag}"] = g["correct"].shift(lag).fillna(0.0)
    return df


# =============================================================================
# 2. CHRONOLOGICAL TRAIN / TEST SPLIT, BY SESSION
# =============================================================================

def split_by_session(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    """Earliest sessions -> train, latest sessions -> test.

    Splitting BY SESSION (not by shuffled trial) matters twice over here: it
    stops trials 30 seconds apart in the same session leaking between train
    and test, and going chronological means "train on the past, test on
    unseen future sessions" -- the only fair way to ask "did the model learn
    something real about this mouse's strategy."
    """
    session_order = (
        df.drop_duplicates("session_id").sort_values("session_idx")["session_id"].tolist()
    )
    n_train_sessions = int(round(train_fraction * len(session_order)))
    train_sessions = set(session_order[:n_train_sessions])

    train_df = df[df["session_id"].isin(train_sessions)].reset_index(drop=True)
    test_df = df[~df["session_id"].isin(train_sessions)].reset_index(drop=True)
    return train_df, test_df

# =============================================================================
# 3. MODELS
# =============================================================================

class ChoiceMLP(nn.Module):
    """Plain feedforward net: one prediction per trial, no memory of any
    other trial except whatever's explicitly concatenated into its input."""

    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,) logits


class ChoiceGRU(nn.Module):
    """Recurrent net: consumes a whole session as a sequence. The hidden
    state `h` is the network's only channel for carrying information from
    trial t-1 (and, transitively, everything before it) into trial t."""

    def __init__(self, n_features, hidden_size=16):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (1, seq_len, n_features) -- one session per forward call.
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1).squeeze(0)  # (seq_len,) logits


def train_mlp(train_df, test_df, feature_cols, label, epochs=40, batch_size=512, lr=1e-2):
    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
    y_train = torch.tensor(train_df["choice_left"].values, dtype=torch.float32)
    X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
    y_test = torch.tensor(test_df["choice_left"].values, dtype=torch.float32)

    model = ChoiceMLP(len(feature_cols))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(X_train)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            logits = model(X_train[idx])
            loss = loss_fn(logits, y_train[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    [{label}] epoch {epoch + 1:>2}/{epochs}  "
                  f"train BCE {epoch_loss / n:.4f}")

    with torch.no_grad():
        test_logits = model(X_test)
        test_prob = torch.sigmoid(test_logits)
        test_loss = loss_fn(test_logits, y_test).item()
        test_acc = ((test_prob > 0.5).float() == y_test).float().mean().item()

    return model, test_prob.numpy(), test_loss, test_acc


def sessions_as_sequences(frame, feature_cols):
    """Group a trial-level dataframe into one (features, target) tensor pair
    per session, sorted by trial order within the session."""
    seqs = []
    for sid, sub in frame.groupby("session_id", sort=False):
        sub = sub.sort_values("trial_in_session")
        X = torch.tensor(sub[feature_cols].values, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(sub["choice_left"].values, dtype=torch.float32)
        seqs.append((sid, sub.index.to_numpy(), X, y))
    return seqs


def train_rnn(train_df, test_df, feature_cols, epochs=15, lr=1e-2):
    train_seqs = sessions_as_sequences(train_df, feature_cols)
    test_seqs = sessions_as_sequences(test_df, feature_cols)

    model = ChoiceGRU(len(feature_cols))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    order = list(range(len(train_seqs)))
    for epoch in range(epochs):
        np.random.shuffle(order)
        epoch_loss, n_trials = 0.0, 0
        for i in order:
            _, _, X, y = train_seqs[i]
            opt.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(y)
            n_trials += len(y)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    [RNN] epoch {epoch + 1:>2}/{epochs}  "
                  f"train BCE {epoch_loss / n_trials:.4f}")

    # Predict on every test trial, writing predictions back into a
    # (row_index -> prob) array aligned with test_df.
    all_probs = np.zeros(len(test_df), dtype=np.float32)
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for _, row_idx, X, y in test_seqs:
            logits = model(X)
            probs = torch.sigmoid(logits)
            all_probs[row_idx] = probs.numpy()
            total_loss += loss_fn(logits, y).item() * len(y)
            total_correct += ((probs > 0.5).float() == y).sum().item()
            total_n += len(y)

    return model, all_probs, total_loss / total_n, total_correct / total_n


# =============================================================================
# 6. BLOCK-SWITCH ADAPTATION CURVES (test sessions only)
# =============================================================================

def block_switch_records(test_df: pd.DataFrame, window_size: int = SWITCH_WINDOW) -> pd.DataFrame:
    """For every block switch into a biased block (probabilityLeft -> 0.2 or
    0.8), collect the 0%-contrast trials in the next `window_size` trials of
    that session. "Chose left" is recoded into "chose the side the NEW block
    favors" so switches in both directions can be averaged on one axis. If
    the RNN (and the mouse) are tracking the block, this rises over
    trials-since-switch; DNN-narrow has no mechanism to do so and must stay
    flat.

    Needs `pred_dnn_narrow` and `pred_rnn` columns on `test_df`.
    """
    records = []  # (trials_since_switch, matches_new_bias) per curve
    for sid, sub in test_df.groupby("session_id", sort=False):
        sub = sub.sort_values("trial_in_session").reset_index(drop=True)
        switch_idx = sub.index[
            (sub["probabilityLeft"] != sub["probabilityLeft"].shift(1))
            & (sub["probabilityLeft"].isin([0.2, 0.8]))
            & (sub.index > 0)
        ]
        for t0 in switch_idx:
            new_bias_favors_left = sub.loc[t0, "probabilityLeft"] == 0.8
            window = sub.iloc[t0:t0 + window_size]
            window = window[window["signed_contrast"] == 0]
            for _, row in window.iterrows():
                since = int(row["trial_in_session"] - sub.loc[t0, "trial_in_session"])
                if new_bias_favors_left:
                    matches_actual = row["choice_left"]
                    matches_narrow = row["pred_dnn_narrow"]
                    matches_rnn = row["pred_rnn"]
                else:
                    matches_actual = 1 - row["choice_left"]
                    matches_narrow = 1 - row["pred_dnn_narrow"]
                    matches_rnn = 1 - row["pred_rnn"]
                records.append((since, matches_actual, matches_narrow, matches_rnn))

    return pd.DataFrame(
        records, columns=["since", "actual", "dnn_narrow", "rnn"]
    )


# =============================================================================
# MAIN: run the whole comparison and draw the figure
# =============================================================================

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="DNN vs RNN on IBL mouse choice data")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="trials CSV from fetch_ibl_data.py")
    parser.add_argument("--fig", type=Path, default=FIG_PATH, help="where to save the figure")
    args = parser.parse_args(argv)
    matplotlib.use("Agg")

    torch.manual_seed(0)
    np.random.seed(0)

    df = add_features(pd.read_csv(args.data))
    print("=" * 78)
    print(f"Loaded {len(df)} trials from {df['session_id'].nunique()} sessions "
          f"({args.data.name}).")
    zero_contrast_frac = (df["signed_contrast"] == 0).mean()
    print(f"Signed contrast built (+ = stim right, - = stim left, 0 = invisible). "
          f"{zero_contrast_frac:.1%} of trials are 0%-contrast -- pure guesswork "
          f"from the stimulus alone, the trials where history matters most.")

    train_df, test_df = split_by_session(df)
    print(f"Train: {train_df['session_id'].nunique()} sessions / {len(train_df)} trials "
          f"(earliest). Test: {test_df['session_id'].nunique()} sessions / {len(test_df)} "
          f"trials (latest, unseen sessions).")
    # =============================================================================
    # 4. TRAIN ALL THREE MODELS
    # =============================================================================
    print("\n--- Training DNN-narrow (input: signed_contrast only) ---")
    _, prob_narrow, loss_narrow, acc_narrow = train_mlp(train_df, test_df, ["signed_contrast"], "DNN-narrow")

    window_cols = (
        ["signed_contrast", "prev_choice_left", "prev_correct"]
        + [f"choice_left_lag{k}" for k in range(2, LAG_K + 1)]
        + [f"correct_lag{k}" for k in range(2, LAG_K + 1)]
    )
    print(f"\n--- Training DNN-window (input: signed_contrast + last {LAG_K} "
          f"trials' choice/outcome, hand-picked window) ---")
    _, prob_window, loss_window, acc_window = train_mlp(train_df, test_df, window_cols, "DNN-window")

    rnn_cols = ["signed_contrast", "prev_choice_left", "prev_correct"]
    print("\n--- Training RNN / GRU (input per step: signed_contrast, "
          "prev_choice, prev_correct -- fed sequentially, memory is learned) ---")
    _, prob_rnn, loss_rnn, acc_rnn = train_rnn(train_df, test_df, rnn_cols)

    test_df["pred_dnn_narrow"] = prob_narrow
    test_df["pred_dnn_window"] = prob_window
    test_df["pred_rnn"] = prob_rnn

    print("\n--- Overall test-set results (unseen sessions) ---")
    print(f"{'Model':<14}{'Test BCE':>10}{'Test Acc':>10}")
    print(f"{'DNN-narrow':<14}{loss_narrow:>10.4f}{acc_narrow:>10.3f}")
    print(f"{'DNN-window':<14}{loss_window:>10.4f}{acc_window:>10.3f}")
    print(f"{'RNN (GRU)':<14}{loss_rnn:>10.4f}{acc_rnn:>10.3f}")
    print(f"Overall accuracy is dominated by the {1 - zero_contrast_frac:.0%} of trials "
          "with real visual evidence, where even DNN-narrow does fine -- the "
          "interesting gap shows up on 0%-contrast trials below.")

    # =============================================================================
    # 5. THE KEY COMPARISON: 0%-CONTRAST TRIALS (no sensory evidence at all)
    # =============================================================================
    zero_c = test_df[test_df["signed_contrast"] == 0].copy()


    def acc(pred_col):
        return ((zero_c[pred_col] > 0.5).astype(float) == zero_c["choice_left"]).mean()


    acc_zero_narrow = acc("pred_dnn_narrow")
    acc_zero_window = acc("pred_dnn_window")
    acc_zero_rnn = acc("pred_rnn")

    print(f"\n--- 0%-contrast test trials only (n={len(zero_c)}) ---")
    print(f"DNN-narrow accuracy: {acc_zero_narrow:.3f}   "
          f"(predicted P(choose left) has only "
          f"{zero_c['pred_dnn_narrow'].nunique()} unique value(s) -- "
          f"it structurally CANNOT vary by trial)")
    print(f"DNN-window accuracy: {acc_zero_window:.3f}")
    print(f"RNN accuracy:        {acc_zero_rnn:.3f}")


    # =========================================================================
    # 6. Block-switch adaptation curves
    # =========================================================================
    switch_df = block_switch_records(test_df)
    curve = switch_df.groupby("since").agg(["mean", "sem", "count"])
    print(f"\n--- Block-switch adaptation (test sessions, 0%-contrast trials "
          f"only, n={len(switch_df)} trial-observations across "
          f"{switch_df['since'].nunique()} post-switch positions) ---")
    print("If a curve rises toward 1.0 as 'trials since switch' grows, that "
          "model is tracking the hidden block. A flat line means it isn't (and,"
          " for DNN-narrow, mathematically can't).")


    # =============================================================================
    # 7. FIGURE
    # =============================================================================
    # Palette (validated categorical order -- see the project's dataviz skill):
    # color follows the MODEL identity and stays consistent across every panel.
    COL_NARROW = "#2a78d6"   # blue   -- DNN-narrow
    COL_WINDOW = "#eb6834"   # orange -- DNN-window
    COL_RNN = "#1baf7a"      # aqua   -- RNN (GRU)
    COL_ACTUAL = "#0b0b0b"   # primary ink -- real mouse behavior (not a model)
    COL_MUTED = "#898781"    # muted ink -- chance / reference lines
    COL_GRID = "#e1e0d9"
    SURFACE = "#fcfcfb"

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#0b0b0b",
        "text.color": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "font.size": 10,
    })

    fig = plt.figure(figsize=(13, 9), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)

    # --- Panel A: architecture schematic ------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("A. Architecture: memory is the only difference", fontsize=11, loc="left")

    box_style = dict(boxstyle="round,pad=0.28", linewidth=1.3)


    def box(x, y, w, h, text, fc, ec="#0b0b0b"):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, fc=fc, ec=ec, **box_style))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


    def arrow(xy_from, xy_to, color="#52514e", style="-", connectionstyle=None):
        ax.add_patch(mpatches.FancyArrowPatch(
            xy_from, xy_to, arrowstyle="-|>", mutation_scale=11,
            linewidth=1.4, color=color, linestyle=style,
            connectionstyle=connectionstyle, shrinkA=2, shrinkB=2,
        ))


    # Shared box layout for both rows: [Input(t-1), Hidden(t-1), Output(t-1)],
    # then a gap, then [Input(t), Hidden(t), Output(t)].
    xs = [0.2, 1.9, 3.6, 5.7, 7.4, 9.1]
    w, h = 1.5, 0.9

    # DNN row (top): two completely independent forward passes -- nothing
    # connects Hidden(t-1) to Hidden(t).
    y_dnn = 6.6
    ax.text(0.2, 8.4, "DNN (feedforward)", fontsize=9.5, fontweight="bold", color=COL_NARROW)
    labels = ["Input\n(t-1)", "Hidden", "Output\n(t-1)", "Input\n(t)", "Hidden", "Output\n(t)"]
    for x, lab in zip(xs, labels):
        box(x, y_dnn, w, h, lab, "#eef2f8")
    for i in (0, 3):
        arrow((xs[i] + w, y_dnn + h / 2), (xs[i + 1], y_dnn + h / 2))
        arrow((xs[i + 1] + w, y_dnn + h / 2), (xs[i + 2], y_dnn + h / 2))
    # The broken link -- the whole point of the diagram.
    ax.plot([xs[1] + w / 2, xs[4] + w / 2], [y_dnn + h + 0.55, y_dnn + h + 0.55],
            color=COL_MUTED, linestyle=":", linewidth=1.3)
    ax.text((xs[1] + xs[4]) / 2 + w / 2, y_dnn + h + 0.7, "no connection",
            ha="center", fontsize=7.5, color=COL_MUTED, style="italic")

    # RNN row (bottom): same layout, but Hidden(t-1) feeds forward into Hidden(t).
    y_rnn = 3.6
    ax.text(0.2, 5.4, "RNN (recurrent)", fontsize=9.5, fontweight="bold", color=COL_RNN)
    for x, lab in zip(xs, labels):
        box(x, y_rnn, w, h, lab, "#e7f7f1")
    for i in (0, 3):
        arrow((xs[i] + w, y_rnn + h / 2), (xs[i + 1], y_rnn + h / 2))
        arrow((xs[i + 1] + w, y_rnn + h / 2), (xs[i + 2], y_rnn + h / 2))
    arrow((xs[1] + w / 2, y_rnn + h), (xs[4] + w / 2, y_rnn + h), color=COL_RNN,
          connectionstyle="arc3,rad=-0.35")
    ax.text((xs[1] + xs[4]) / 2 + w / 2, y_rnn + h + 0.75,
            "hidden state carried forward", ha="center", fontsize=7.5,
            color=COL_RNN, style="italic")

    ax.text(0.2, 1.9,
            "A DNN fed only the current trial has no input that could tell it\n"
            "which hidden block is active. An RNN's hidden state can.",
            fontsize=8, color="#52514e")

    # --- Panel B: overall test accuracy --------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    models = ["DNN-narrow", "DNN-window", "RNN (GRU)"]
    accs = [acc_narrow, acc_window, acc_rnn]
    colors = [COL_NARROW, COL_WINDOW, COL_RNN]
    bars = ax.bar(models, accs, color=colors, width=0.55, zorder=3)
    ax.axhline(0.5, color=COL_MUTED, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(-0.42, 0.52, "chance", fontsize=8, color=COL_MUTED, ha="left")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.3f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test accuracy (all trials)")
    ax.set_title("B. Overall accuracy -- close, because most trials\nhave real "
                 "visual evidence", fontsize=11, loc="left")
    ax.grid(axis="y", color=COL_GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Panel C: 0%-contrast accuracy (the real test) ------------------------
    ax = fig.add_subplot(gs[1, 0])
    accs0 = [acc_zero_narrow, acc_zero_window, acc_zero_rnn]
    bars = ax.bar(models, accs0, color=colors, width=0.55, zorder=3)
    ax.axhline(0.5, color=COL_MUTED, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(-0.42, 0.52, "chance", fontsize=8, color=COL_MUTED, ha="left")
    for b, a in zip(bars, accs0):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.3f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test accuracy (0%-contrast trials only)")
    ax.set_title(f"C. Zero-contrast trials (n={len(zero_c)}): the stimulus gives\n"
                 "zero information -- only history can help", fontsize=11, loc="left")
    ax.grid(axis="y", color=COL_GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Panel D: block-switch adaptation curves ------------------------------
    ax = fig.add_subplot(gs[1, 1])
    since_vals = curve.index.to_numpy()


    def plot_curve(col, color, label, linestyle="-"):
        mean = curve[(col, "mean")].to_numpy()
        sem = curve[(col, "sem")].fillna(0).to_numpy()
        ax.plot(since_vals, mean, color=color, linewidth=2, label=label, linestyle=linestyle)
        ax.fill_between(since_vals, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)


    plot_curve("actual", COL_ACTUAL, "Real mouse")
    plot_curve("rnn", COL_RNN, "RNN (GRU)")
    plot_curve("dnn_narrow", COL_NARROW, "DNN-narrow")
    ax.axhline(0.5, color=COL_MUTED, linestyle="--", linewidth=1.2)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("Trials since block switch (0%-contrast trials only)")
    ax.set_ylabel("P(choice matches new block)")
    ax.set_title("D. Adapting to a silent block switch: RNN tracks it,\n"
                  "DNN-narrow structurally can't", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(color=COL_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("DNN vs RNN on real IBL mouse choice data: only recurrence can "
                 "track a hidden, drifting bias", fontsize=13, y=0.99)
    fig.savefig(args.fig, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nSaved figure to {args.fig}")


if __name__ == "__main__":
    main()
