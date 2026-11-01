# Data license and attribution

`data/ibl_trials.csv`, and the figures and tables derived from it
(`rnn_vs_dnn_ibl.png` and the saved outputs in `explore_ibl_data.ipynb`),
are based on data from the **International Brain Laboratory (IBL)** public
data release, served through OpenAlyx
(<https://openalyx.internationalbrainlab.org>).

The IBL shares this data under the **Creative Commons Attribution 4.0
International** license (CC BY 4.0):
<https://creativecommons.org/licenses/by/4.0/>. The same license applies to
`data/ibl_trials.csv` as redistributed here.

## What was selected and changed

- **Selection:** subject `ibl_witten_12`, project
  `ibl_neuropixel_brainwide_01`, task protocol `biasedChoiceWorld`; the 40
  earliest sessions (2019-07-18 to 2019-09-19).
- **Columns:** from each session's `_ibl_trials.table.pqt`, only `choice`,
  `contrastLeft`, `contrastRight`, `feedbackType`, `probabilityLeft`,
  `goCue_times`, `firstMovement_times` and `response_times` were kept.
- **Rows:** no-go trials (`choice == 0`, 119 in total) were removed.
- **Added:** session id, session index, session start time and trial number
  within the session.

`fetch_ibl_data.py` regenerates the file from the IBL database.

## Crediting the IBL

If you reuse this data, credit the International Brain Laboratory and link
the license above. The behavioral task is described in:

> The International Brain Laboratory et al. (2021). Standardized and
> reproducible measurement of decision-making in mice. *eLife* 10:e63711.
> <https://doi.org/10.7554/eLife.63711>

The IBL's own documentation explains how to cite specific data releases.
