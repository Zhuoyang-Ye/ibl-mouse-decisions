"""fetch_ibl_data.load_session_trials against a fake ONE (no network).
Needs ONE-api installed (the `ibl` env); skipped elsewhere."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("one.api")
import fetch_ibl_data as fetch  # noqa: E402

RAW = pd.DataFrame({
    "choice": [1, 0, -1],  # the middle trial is a no-go
    "contrastLeft": [1.0, np.nan, np.nan],
    "contrastRight": [np.nan, 0.25, 0.0],
    "feedbackType": [1, -1, 1],
    "probabilityLeft": [0.5, 0.5, 0.5],
    "goCue_times": [1.0, 2.0, 3.0],
    "response_times": [1.3, 2.4, 3.2],
    "intervals_0": [0.5, 1.5, 2.5],  # an extra column that must be dropped
})


class FakeOne:
    def __init__(self, datasets, table=None):
        self.datasets, self.table = datasets, table

    def list_datasets(self, eid):
        return self.datasets

    def load_dataset(self, eid, name, collection=None):
        assert (name, collection) == ("_ibl_trials.table.pqt", "alf/task_00")
        return self.table.copy()

    def load_object(self, eid, obj, collection=None):
        assert (obj, collection) == ("trials", "alf")
        return type("AlfBunch", (), {"to_df": lambda self: RAW.copy()})()


ID_COLUMNS = ["session_id", "session_idx", "session_start_time", "trial_in_session"]


def test_session_with_trials_table():
    table = RAW.assign(firstMovement_times=[1.1, np.nan, 3.05])
    one = FakeOne(["alf/task_00/_ibl_trials.table.pqt", "alf/task_00/_ibl_wheel.position.npy"],
                  table)
    trials, n_nogo = fetch.load_session_trials(one, "eid1", 3, "2020-01-01T10:00:00")
    assert n_nogo == 1
    assert list(trials.columns) == ID_COLUMNS + fetch.KEEP_COLUMNS
    assert trials["trial_in_session"].tolist() == [0, 1]
    assert (trials["session_idx"] == 3).all()
    assert trials["firstMovement_times"].tolist() == [1.1, 3.05]


def test_session_with_one_file_per_attribute():
    one = FakeOne(["alf/_ibl_trials.choice.npy", "alf/_ibl_trials.goCue_times.npy"])
    trials, n_nogo = fetch.load_session_trials(one, "eid2", 0, "2020-01-02T10:00:00")
    assert n_nogo == 1 and len(trials) == 2
    assert list(trials.columns) == ID_COLUMNS + fetch.KEEP_COLUMNS
    # this session has no first-movement times: the column exists but is empty
    assert trials["firstMovement_times"].isna().all()
    assert trials["response_times"].tolist() == [1.3, 3.2]


def test_session_without_trials_gives_a_clear_error():
    with pytest.raises(ValueError, match="no trials data"):
        fetch.load_session_trials(FakeOne(["alf/_ibl_wheel.position.npy"]), "eid3", 0, "x")
