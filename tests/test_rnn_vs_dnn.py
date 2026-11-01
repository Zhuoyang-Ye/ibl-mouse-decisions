import numpy as np
import pandas as pd
import pytest
import torch

import rnn_vs_dnn_ibl as ibl


def make_trials(n_sessions=4, n_trials=30, seed=0):
    """Tiny synthetic stand-in for data/ibl_trials.csv."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        for t in range(n_trials):
            side_left = rng.random() < 0.5
            contrast = rng.choice([0.0, 0.25, 1.0])
            rows.append({
                "session_id": f"sess{s}",
                "session_idx": s,
                "trial_in_session": t,
                "contrastLeft": contrast if side_left else np.nan,
                "contrastRight": np.nan if side_left else contrast,
                "choice": rng.choice([1, -1]),
                "feedbackType": rng.choice([1, -1]),
                "probabilityLeft": 0.8 if t >= n_trials // 2 else 0.2,
            })
    return pd.DataFrame(rows)


def test_signed_contrast_and_targets():
    df = ibl.add_features(make_trials())
    left = df["contrastLeft"].notna()
    assert (df.loc[left, "signed_contrast"] <= 0).all()
    assert (df.loc[~left, "signed_contrast"] >= 0).all()
    assert set(df["choice_left"].unique()) <= {0.0, 1.0}


def test_history_features_do_not_leak_across_sessions():
    df = ibl.add_features(make_trials(), lag_k=3)
    first_trials = df[df["trial_in_session"] == 0]
    assert (first_trials["prev_choice_left"] == 0).all()
    assert (first_trials["choice_left_lag3"] == 0).all()
    # Within a session, prev_choice_left is exactly the previous trial's choice.
    s0 = df[df["session_id"] == "sess0"].reset_index(drop=True)
    assert (s0["prev_choice_left"].iloc[1:].to_numpy() == s0["choice_left"].iloc[:-1].to_numpy()).all()


def test_split_is_chronological_and_disjoint():
    df = ibl.add_features(make_trials(n_sessions=5))
    train, test = ibl.split_by_session(df, train_fraction=0.8)
    assert set(train["session_id"]).isdisjoint(test["session_id"])
    assert train["session_idx"].max() < test["session_idx"].min()
    assert len(train) + len(test) == len(df)


def test_models_output_shapes():
    assert ibl.ChoiceMLP(3)(torch.randn(7, 3)).shape == (7,)
    assert ibl.ChoiceGRU(3)(torch.randn(1, 12, 3)).shape == (12,)


def test_training_smoke():
    torch.manual_seed(0)
    df = ibl.add_features(make_trials())
    train, test = ibl.split_by_session(df)
    _, probs, loss, acc = ibl.train_mlp(train, test, ["signed_contrast"], "t", epochs=1)
    assert probs.shape == (len(test),) and 0 <= acc <= 1 and np.isfinite(loss)
    _, probs, loss, acc = ibl.train_rnn(train, test, ["signed_contrast", "prev_choice_left"], epochs=1)
    assert probs.shape == (len(test),) and np.all((probs >= 0) & (probs <= 1))


def test_block_switch_records_recode_to_new_bias():
    df = ibl.add_features(make_trials(n_sessions=1, n_trials=40))
    df["pred_dnn_narrow"] = 0.5
    df["pred_rnn"] = df["choice_left"]          # a "perfect" model
    records = ibl.block_switch_records(df, window_size=10)
    # One switch (0.2 -> 0.8 at trial 20); only 0%-contrast trials are kept.
    expected = df[(df["trial_in_session"].between(20, 29)) & (df["signed_contrast"] == 0)]
    assert len(records) == len(expected)
    assert (records["since"].between(0, 9)).all()
    assert (records["dnn_narrow"] == 0.5).all()
    assert (records["rnn"] == records["actual"]).all()
