"""
Fetch real mouse decision-making data from the International Brain Laboratory
==============================================================================

WHAT THIS SCRIPT IS FOR
------------------------
Stage 1 of a two-stage pipeline. This script has ONE job: talk to the public
IBL database (via the `one-api` / `ibllib` packages) and cache a clean trial
table to `data/ibl_trials.csv`. It does NOT build or train any models -- that
happens in `rnn_vs_dnn_ibl.py`, which only needs pandas/numpy and can run in
a totally different Python environment (no network access, no ibllib).

Why split it this way? `one-api`/`ibllib` and `torch` don't currently live in
the same conda env on this machine (the `ibl` env has the former, a torch
env such as `code4fun` has the latter). Rather than fight environment
conflicts, the pipeline is just:

    conda run -n ibl       python fetch_ibl_data.py       # this file
    conda run -n code4fun  python rnn_vs_dnn_ibl.py        # modeling + plots

THE DATA
--------
The International Brain Laboratory (https://www.internationalbrainlab.org)
is a consortium that published a fully open mouse decision-making dataset:
on each trial, a visual grating appears on the LEFT or RIGHT of a screen at
one of several contrasts (0-100%), and the mouse turns a wheel to indicate
which side it saw. Critically, the *prior probability* that the stimulus
will appear on the left (`probabilityLeft`) is held fixed at 0.5 for a
while, then silently switched to 0.2 or 0.8 for a block of ~50-100 trials,
then switched again -- and this is never signaled to the mouse. Well-trained
mice learn to track the current block from the recent history of stimuli,
choices and rewards, and bias their choices accordingly -- especially on
0%-contrast trials where the stimulus itself carries no information at all.

That block-tracking behavior is exactly the kind of thing a memoryless model
structurally cannot do, and a recurrent model can. See `rnn_vs_dnn_ibl.py`
for the actual DNN vs RNN comparison built on top of this data.

We use subject `ibl_witten_12` from the IBL Brain-Wide Map project: it has
80+ sessions of the standard biased task with no optogenetics or task
manipulations, i.e. "clean" behavior -- a good subject for a first pass.

Run this file with:
    conda run -n ibl python fetch_ibl_data.py
It prints progress to the console and writes `data/ibl_trials.csv` next to
this script. Safe to re-run; it just overwrites the cache.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from one.api import ONE

OPENALYX_URL = "https://openalyx.internationalbrainlab.org"
# The IBL's shared read-only account. Newer ONE-api versions no longer fill
# it in by themselves, and the IBL plans to retire it: to use your own (free)
# account, set IBL_ALYX_USERNAME and IBL_ALYX_PASSWORD.
PUBLIC_USERNAME, PUBLIC_PASSWORD = "intbrainlab", "international"
SUBJECT = "ibl_witten_12"
PROJECT = "ibl_neuropixel_brainwide_01"
TASK_PROTOCOL = "biasedChoiceWorld"
MAX_SESSIONS = 40  # cap runtime; ~80 exist for this subject, 40 is plenty of data
OUT_PATH = Path(__file__).parent / "data" / "ibl_trials.csv"

# Columns we actually need downstream. The models only ever see
# stimulus/choice/outcome; the three timing columns (seconds, on each
# session's own clock) are kept for reaction-time plots in
# explore_ibl_data.ipynb and ibl_explorer.py.
KEEP_COLUMNS = [
    "choice",
    "contrastLeft",
    "contrastRight",
    "feedbackType",
    "probabilityLeft",
    "goCue_times",
    "firstMovement_times",
    "response_times",
]


def connect():
    """Read-only connection to the public IBL database, with the account in
    IBL_ALYX_USERNAME / IBL_ALYX_PASSWORD if set, else the shared one."""
    username = os.environ.get("IBL_ALYX_USERNAME")
    if username:
        password = os.environ.get("IBL_ALYX_PASSWORD")
    else:
        username, password = PUBLIC_USERNAME, PUBLIC_PASSWORD
    return ONE(base_url=OPENALYX_URL, username=username, password=password, silent=True)


def find_sessions(one, subject, task_protocol=TASK_PROTOCOL, project=PROJECT):
    """All of a subject's sessions as (start_time, eid) pairs, oldest first.

    Pass project=None to search across every project.
    """
    query = {"task_protocol": task_protocol, "subject": subject}
    if project:
        query["projects"] = project
    eids, details = one.search(details=True, **query)
    return sorted(zip([d["start_time"] for d in details], [str(e) for e in eids]))


def load_session_trials(one, eid, session_idx, start_time):
    """One session's trials, KEEP_COLUMNS only, no-go trials removed.

    Returns (trials, n_dropped_nogo).
    """
    # The trials live under different "collection" subfolders depending on
    # which pipeline version processed the session, so we look up the exact
    # path per session rather than hardcoding one.
    datasets = one.list_datasets(eid)
    tables = [d for d in datasets if d.endswith("_ibl_trials.table.pqt")]
    if tables:
        collection = tables[0].rsplit("/", 1)[0]
        trials = one.load_dataset(eid, "_ibl_trials.table.pqt", collection=collection)
    else:
        # Some sessions store each trial attribute as its own .npy file
        # (_ibl_trials.choice.npy, ...) instead of one table.
        files = [d for d in datasets if "_ibl_trials." in d]
        if not files:
            raise ValueError(f"session {eid} has no trials data")
        collection = files[0].rsplit("/", 1)[0] if "/" in files[0] else None
        trials = one.load_object(eid, "trials", collection=collection).to_df()

    # Columns a session doesn't have (some lack firstMovement_times) become NaN.
    trials = trials.reindex(columns=KEEP_COLUMNS)

    # choice == 0 means the mouse never responded (a "no-go" / omission
    # trial) -- there's no left/right label to predict, so drop it.
    n_dropped_nogo = int((trials["choice"] == 0).sum())
    trials = trials[trials["choice"] != 0].reset_index(drop=True)

    trials.insert(0, "trial_in_session", np.arange(len(trials)))
    trials.insert(0, "session_start_time", start_time)
    trials.insert(0, "session_idx", session_idx)
    trials.insert(0, "session_id", eid)
    return trials, n_dropped_nogo


def main():
    print("=" * 78)
    print(f"Connecting to the public IBL database (subject={SUBJECT})...")
    one = connect()

    all_sessions = find_sessions(one, SUBJECT)
    sessions = all_sessions[:MAX_SESSIONS]
    print(f"Found {len(all_sessions)} sessions total; using the earliest {len(sessions)} "
          f"(chronological order matters -- we'll train on the past and test "
          f"on the future).")

    all_frames = []
    n_dropped_nogo = 0
    t_start = time.time()

    for session_idx, (start_time, eid) in enumerate(sessions):
        trials, n_dropped = load_session_trials(one, eid, session_idx, start_time)
        n_dropped_nogo += n_dropped
        all_frames.append(trials)

        elapsed = time.time() - t_start
        print(f"  [{session_idx + 1:>2}/{len(sessions)}] {start_time[:10]}  "
              f"eid={eid[:8]}...  {len(trials):>4} trials  "
              f"(elapsed {elapsed:5.1f}s)")

    df = pd.concat(all_frames, ignore_index=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("=" * 78)
    print(f"Dropped {n_dropped_nogo} no-go trials (choice == 0) across all sessions.")
    print(f"Wrote {len(df)} trials from {len(sessions)} sessions to {OUT_PATH}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"probabilityLeft block values seen: {sorted(df['probabilityLeft'].unique())}")
    print("Done. You can now run rnn_vs_dnn_ibl.py in a torch-enabled env, e.g.:")
    print("    conda run -n code4fun python rnn_vs_dnn_ibl.py")


if __name__ == "__main__":
    main()
