import numpy as np
import pandas as pd
import pytest

import ibl_analysis as ia


def make_trials():
    """Two sessions with a known block structure and a mouse that always
    chooses the side its current block favors (alternates in 0.5 blocks).

    session 0: 0.5 x4, 0.8 x5, 0.2 x6, 0.8 x3   (switches 0.8->0.2, 0.2->0.8)
    session 1: 0.5 x3, 0.2 x4                    (no biased<->biased switch)
    """
    rows = []
    for s, blocks in enumerate([[(0.5, 4), (0.8, 5), (0.2, 6), (0.8, 3)],
                                [(0.5, 3), (0.2, 4)]]):
        t = 0
        for p_left, n in blocks:
            for _ in range(n):
                stim_right = t % 2 == 0
                if p_left == 0.5:
                    chose_right = t % 2 == 1
                else:
                    chose_right = p_left == 0.2
                rows.append({
                    "session_id": f"sess{s}",
                    "session_idx": s,
                    "session_start_time": f"2020-01-0{s + 1}T10:00:00",
                    "trial_in_session": t,
                    "choice": -1 if chose_right else 1,
                    "contrastLeft": np.nan if stim_right else 0.0,
                    "contrastRight": 0.0 if stim_right else np.nan,
                    "feedbackType": 1 if chose_right == stim_right else -1,
                    "probabilityLeft": p_left,
                    "goCue_times": t + 0.0,
                    "firstMovement_times": t + 0.1,
                    "response_times": t + (0.3 if t % 3 else -0.5),
                })
                t += 1
    return pd.DataFrame(rows)


@pytest.fixture
def df():
    return ia.prepare(make_trials())


def test_choice_and_contrast_conventions(df):
    assert (df["chose_right"] == (df["choice"] == -1)).all()
    # 0 % trials still have a side, taken from which contrast column is set
    assert (df["stim_right"] == df["contrastRight"].notna()).all()
    assert (df["signed_contrast"] == 0).all()


def test_block_columns(df):
    s0 = df[df["session_idx"] == 0]
    assert s0.groupby("block_id")["block_len"].first().tolist() == [4, 5, 6, 3]
    assert s0["trial_in_block"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2]
    first_block_s1 = df[(df["session_idx"] == 1) & (df["trial_in_block"] == 0)].iloc[0]
    # session 1's first block must not see session 0's last block as "previous"
    assert np.isnan(first_block_s1["prev_block_p"])
    assert np.isnan(s0["next_block_p"].iloc[-1])


def test_previous_trial_does_not_cross_sessions(df):
    first = df[df["trial_in_session"] == 0]
    assert first["prev_chose_right"].isna().all()
    assert first["prev_correct"].isna().all()
    rest = df[df["trial_in_session"] > 0]
    assert rest["prev_chose_right"].notna().all()


def test_filtering_keeps_derived_columns(df):
    view = ia.filter_trials(df, blocks=[0.2, 0.8], skip_first=1)
    pd.testing.assert_series_equal(view["prev_chose_right"],
                                   df.loc[view.index, "prev_chose_right"])
    assert set(view["probabilityLeft"]) == {0.2, 0.8}
    assert (view["trial_in_session"] >= 1).all()


def test_block_switch_curve_alignment(df):
    curve = ia.block_switch_curve(df, before=2, after=2, split=False)
    # only the two 0.2<->0.8 switches in session 0 count
    assert curve["offset"].tolist() == [-2, -1, 0, 1]
    assert (curve["n"] == 2).all()
    # this mouse follows each block perfectly
    assert curve.set_index("offset")["p"].to_dict() == {-2: 0.0, -1: 0.0, 0: 1.0, 1: 1.0}


def test_block_switch_curve_stops_at_block_end(df):
    curve = ia.block_switch_curve(df, before=1, after=10, split=True)
    into_left = curve[curve["new_block"] == 0.8]  # last block, 3 trials long
    into_right = curve[curve["new_block"] == 0.2]  # 6 trials long
    assert into_left["offset"].max() == 2
    assert into_right["offset"].max() == 5


def test_block_runs_drop_truncated_last_block(df):
    runs = ia.block_runs(df)
    # session 0: 0.8 (5) and 0.2 (6) are complete; its final 0.8 block and
    # session 1's only biased block are cut off by the session end
    assert runs["length"].tolist() == [5, 6]
    assert len(ia.block_runs(df, drop_last=False)) == 4


def test_psychometric_and_history_counts(df):
    psy = ia.psychometric(df)
    assert psy["n"].sum() == len(df)
    assert psy["p_right"].between(0, 1).all()

    hist = ia.history_table(df, contrast=0.0)
    assert hist["n"].sum() == df["prev_chose_right"].notna().sum()
    assert set(hist["previous"]) <= set(ia.HISTORY_ORDER)


def test_session_summary(df):
    summary = ia.session_summary(df)
    assert summary["n_trials"].tolist() == [18, 7]
    # a perfect block follower
    assert (summary["follows_block_0"] == 1.0).all()


def test_timing_columns_and_chronometric(df):
    assert np.allclose(df["movement_time"], 0.1)
    chrono = ia.chronometric(df, "response_time")
    # non-positive response times are left out
    assert chrono["n"].sum() == (df["response_time"] > 0).sum()
    assert np.allclose(chrono["median"], 0.3)


def test_prepare_without_timing_columns():
    raw = make_trials().drop(columns=["goCue_times", "firstMovement_times", "response_times"])
    out = ia.prepare(raw)
    assert "response_time" not in out and "movement_time" not in out
    assert "median_response_time" not in ia.session_summary(out)
