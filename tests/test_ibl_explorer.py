"""GUI smoke test. Runs only where PyQt5 + pyqtgraph + ONE-api exist (the
`ibl` env); skipped in the torch env and in CI."""

import os

import pytest

pytest.importorskip("pyqtgraph")
pytest.importorskip("one.api")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ibl_explorer as ex  # noqa: E402


@pytest.fixture(scope="module")
def window():
    app = ex.pg.mkQApp()
    w = ex.Explorer(ex.DEFAULT_CSV)
    w.show()
    yield w, app
    w.close()


def test_every_tab_draws(window):
    w, app = window
    for i in range(w.tabs.count()):
        w.tabs.setCurrentIndex(i)
        app.processEvents()
    assert not w.dirty
    assert w.table_model.rowCount() == len(w.view)


def test_filters_that_empty_the_view_do_not_crash(window):
    w, app = window
    w.check_all_sessions(False)
    for i in range(w.tabs.count()):
        w.tabs.setCurrentIndex(i)
        app.processEvents()
    assert len(w.view) == 0
    w.check_all_sessions(True)
    assert len(w.view) == len(w.data)
