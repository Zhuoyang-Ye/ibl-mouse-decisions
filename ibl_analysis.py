"""
Plotting-free analysis helpers for the IBL trials table
=======================================================

Shared by `explore_ibl_data.ipynb` and `ibl_explorer.py` so the notebook and
the GUI always compute the same numbers. Needs only numpy + pandas.

Workflow:

    df = prepare(pd.read_csv("data/ibl_trials.csv"))   # derive columns ONCE
    view = filter_trials(df, sessions=..., blocks=..., skip_first=...)
    psychometric(view), block_switch_curve(view), ...

`prepare` computes everything that depends on trial order (previous trial,
block boundaries, position in block) on the full, unfiltered table. Every
other function only groups rows, so it stays correct on any row subset:
dropping the 0.5 block or a few sessions never shifts "the previous trial".

Conventions (IBL ALF):
- `choice == -1` means the mouse reported RIGHT, `+1` LEFT (no-go trials were
  already dropped by fetch_ibl_data.py).
- Contrasts are fractions (0.0625 = 6.25 %); exactly one of
  contrastLeft/contrastRight is set per trial, which also gives 0 % trials a
  "correct side".
- `probabilityLeft` 0.8 = left block, 0.2 = right block, 0.5 = the unbiased
  block that opens every session (first 90 trials).
"""

import numpy as np
import pandas as pd

LEFT_BLOCK, NEUTRAL_BLOCK, RIGHT_BLOCK = 0.8, 0.5, 0.2
BIASED_BLOCKS = (RIGHT_BLOCK, LEFT_BLOCK)
EASY_CONTRAST = 1.0  # |contrast| >= this counts as an "easy" trial

BLOCK_LABELS = {
    LEFT_BLOCK: "Left block (pLeft 0.8)",
    NEUTRAL_BLOCK: "Unbiased (pLeft 0.5)",
    RIGHT_BLOCK: "Right block (pLeft 0.2)",
}
HISTORY_ORDER = ["Left, rewarded", "Left, error", "Right, rewarded", "Right, error"]

# Timing measures, all relative to the go cue (= stimulus onset in this task).
RT_MEASURES = {
    "response_time": "Response time (response - go cue)",
    "movement_time": "First-movement time (first movement - go cue)",
}

# One color meaning per side, everywhere: left / left block = blue, right /
# right block = red (the palette's diverging pair), unbiased = neutral gray.
# Mouse behavior that has no side is drawn in ink.
COLORS = {
    "left": "#2a78d6",
    "right": "#e34948",
    "neutral": "#898781",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}
BLOCK_COLORS = {
    LEFT_BLOCK: COLORS["left"],
    NEUTRAL_BLOCK: COLORS["neutral"],
    RIGHT_BLOCK: COLORS["right"],
}


def _sem(p, n):
    """Binomial standard error of a proportion."""
    return np.sqrt(p * (1 - p) / n)


def prepare(trials: pd.DataFrame) -> pd.DataFrame:
    """Add derived per-trial columns. Call once, on the full table.

    Added columns:
      signed_contrast   right-positive contrast (fraction), abs_contrast
      stim_right        stimulus was on the right (also defined at 0 %)
      chose_right, correct
      prev_chose_right, prev_correct   1.0 / 0.0, NaN on a session's 1st trial
      block_id          unique id of each contiguous run of probabilityLeft
      trial_in_block    0 = first trial of the block
      block_len         full length of the trial's block
      prev_block_p, next_block_p   neighboring blocks' probabilityLeft in the
                        same session (NaN at session edges)
      date              session date (YYYY-MM-DD)
      response_time, movement_time   seconds after the go cue (if the timing
                        columns are present)
    """
    df = trials.sort_values(["session_idx", "trial_in_session"]).reset_index(drop=True)

    df["signed_contrast"] = df["contrastRight"].fillna(0) - df["contrastLeft"].fillna(0)
    df["abs_contrast"] = df["signed_contrast"].abs()
    df["stim_right"] = df["contrastRight"].notna()
    df["chose_right"] = df["choice"] == -1
    df["correct"] = df["feedbackType"] == 1
    df["date"] = df["session_start_time"].str[:10]

    by_session = df.groupby("session_id", sort=False)
    df["prev_chose_right"] = by_session["chose_right"].shift(1).astype(float)
    df["prev_correct"] = by_session["correct"].shift(1).astype(float)

    # A block starts wherever probabilityLeft changes, or a session starts.
    starts = df["probabilityLeft"].ne(by_session["probabilityLeft"].shift(1))
    df["block_id"] = starts.cumsum() - 1
    by_block = df.groupby("block_id")
    df["trial_in_block"] = by_block.cumcount()
    df["block_len"] = by_block["choice"].transform("size")

    blocks = by_block.agg(session_id=("session_id", "first"), p=("probabilityLeft", "first"))
    same_prev = blocks["session_id"].eq(blocks["session_id"].shift(1))
    same_next = blocks["session_id"].eq(blocks["session_id"].shift(-1))
    df["prev_block_p"] = df["block_id"].map(blocks["p"].shift(1).where(same_prev))
    df["next_block_p"] = df["block_id"].map(blocks["p"].shift(-1).where(same_next))

    if {"goCue_times", "response_times"} <= set(df.columns):
        df["response_time"] = df["response_times"] - df["goCue_times"]
    if {"goCue_times", "firstMovement_times"} <= set(df.columns):
        df["movement_time"] = df["firstMovement_times"] - df["goCue_times"]
    return df


def filter_trials(df, sessions=None, blocks=None, skip_first=0):
    """Row subset of a prepared table.

    sessions: iterable of session_idx to keep (None = all)
    blocks: iterable of probabilityLeft values to keep (None = all)
    skip_first: drop each session's first N trials (90 = the unbiased opener)
    """
    keep = df["trial_in_session"] >= skip_first
    if sessions is not None:
        keep &= df["session_idx"].isin(list(sessions))
    if blocks is not None:
        keep &= df["probabilityLeft"].isin(list(blocks))
    return df[keep]


def session_summary(df):
    """One row per session: trial count, accuracy, and how much the mouse
    follows the hidden block on 0 % trials (0.5 = ignores it)."""
    zero_biased = df["abs_contrast"].eq(0) & df["probabilityLeft"].isin(BIASED_BLOCKS)
    follows_block = df["chose_right"] == (df["probabilityLeft"] == RIGHT_BLOCK)
    correct = df["correct"].astype(float)
    tmp = pd.DataFrame({
        "session_idx": df["session_idx"],
        "date": df["date"],
        "correct": correct,
        "easy_correct": correct.where(df["abs_contrast"] >= EASY_CONTRAST),
        "follows_block_0": follows_block.astype(float).where(zero_biased),
    })
    aggs = {
        "n_trials": ("correct", "size"),
        "accuracy": ("correct", "mean"),
        "easy_accuracy": ("easy_correct", "mean"),
        "follows_block_0": ("follows_block_0", "mean"),
    }
    if "response_time" in df:
        tmp["response_time"] = df["response_time"]
        aggs["median_response_time"] = ("response_time", "median")
    return tmp.groupby(["session_idx", "date"]).agg(**aggs).reset_index()


def psychometric(df, by="probabilityLeft"):
    """P(chose right) at each signed contrast, per level of `by` (None = pooled)."""
    keys = [by, "signed_contrast"] if by else ["signed_contrast"]
    out = df.groupby(keys)["chose_right"].agg(n="size", p_right="mean").reset_index()
    out["sem"] = _sem(out["p_right"], out["n"])
    return out


def block_switch_curve(df, before=10, after=40, max_contrast=None, split=True):
    """P(choice toward the NEW block's side) aligned to block switches.

    Only switches between two biased blocks count (0.2 <-> 0.8). offset 0 is
    the first trial of the new block; negative offsets are the last trials of
    the old block. Trials past the new block's end, or before the old
    block's start, are not included.

    max_contrast: keep only trials with |contrast| <= this (None = all).
    split: one curve per new block value (True) or pooled (False).
    """
    biased = df["probabilityLeft"].isin(BIASED_BLOCKS)
    in_new = biased & df["prev_block_p"].isin(BIASED_BLOCKS) & (df["trial_in_block"] < after)
    to_end = df["trial_in_block"] - df["block_len"]
    in_old = biased & df["next_block_p"].isin(BIASED_BLOCKS) & (to_end >= -before)

    aligned = pd.concat([
        pd.DataFrame({"offset": df.loc[in_new, "trial_in_block"],
                      "new_block": df.loc[in_new, "probabilityLeft"]}),
        pd.DataFrame({"offset": to_end[in_old],
                      "new_block": df.loc[in_old, "next_block_p"]}),
    ])
    aligned["chose_right"] = df.loc[aligned.index, "chose_right"].to_numpy()
    aligned["abs_contrast"] = df.loc[aligned.index, "abs_contrast"].to_numpy()
    if max_contrast is not None:
        aligned = aligned[aligned["abs_contrast"] <= max_contrast]
    aligned["chose_new_side"] = aligned["chose_right"] == (aligned["new_block"] == RIGHT_BLOCK)

    keys = ["new_block", "offset"] if split else ["offset"]
    out = aligned.groupby(keys)["chose_new_side"].agg(n="size", p="mean").reset_index()
    out["sem"] = _sem(out["p"], out["n"])
    return out


def history_table(df, contrast=0.0):
    """P(chose right) at one |contrast|, by previous trial (side x outcome)
    and current block. Splitting by block matters: "previous choice right"
    is itself more common in right blocks, so pooling mixes the two effects."""
    sub = df[df["abs_contrast"].eq(contrast) & df["prev_chose_right"].notna()]
    side = np.where(sub["prev_chose_right"] == 1, "Right", "Left")
    outcome = np.where(sub["prev_correct"] == 1, "rewarded", "error")
    previous = pd.Categorical(pd.Series(side, index=sub.index) + ", " + outcome,
                              categories=HISTORY_ORDER, ordered=True)
    out = (sub.assign(previous=previous)
           .groupby(["previous", "probabilityLeft"], observed=True)["chose_right"]
           .agg(n="size", p_right="mean").reset_index())
    out["sem"] = _sem(out["p_right"], out["n"])
    return out


def block_runs(df, biased_only=True, drop_last=True):
    """One row per block: session, probabilityLeft, full length.

    drop_last removes each session's final block, which the end of the
    session cut short (keeping it biases lengths downward).
    """
    runs = df.groupby("block_id").agg(
        session_idx=("session_idx", "first"),
        probabilityLeft=("probabilityLeft", "first"),
        length=("block_len", "first"),
        is_last=("next_block_p", lambda s: s.isna().all()),
    ).reset_index()
    if biased_only:
        runs = runs[runs["probabilityLeft"].isin(BIASED_BLOCKS)]
    if drop_last:
        runs = runs[~runs["is_last"]]
    return runs.reset_index(drop=True)


def chronometric(df, measure="response_time"):
    """Median and quartiles of a timing measure per |contrast| x correct.
    Trials with a missing or non-positive time are left out."""
    sub = df[df[measure] > 0]
    out = (sub.groupby(["abs_contrast", "correct"])[measure]
           .agg(n="size", median="median",
                q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
           .reset_index())
    return out
