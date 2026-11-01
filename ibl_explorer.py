"""
IBL trial explorer: a desktop GUI for playing with the trials table
===================================================================

Run it in the `ibl` conda env (PyQt5 and pyqtgraph come with ibllib):

    conda activate ibl
    python ibl_explorer.py               # from the repo root; opens data/ibl_trials.csv
    python ibl_explorer.py other.csv     # any CSV written by fetch_ibl_data.py

On Windows without Python, use the built app instead ("IBL Explorer.exe",
see the README); ibl_explorer.spec is its PyInstaller recipe.

LEFT PANEL
  Data      open/save a trials CSV, or download any subject's sessions from
            the public IBL database (same code path as fetch_ibl_data.py)
  Filter    which sessions, which blocks, and how many opening trials of
            each session to skip (90 = the unbiased 0.5 block)

TABS (each redraws from the filtered trials)
  Sessions       trial count, accuracy and block-following per session
  Psychometric   P(right) vs signed contrast, one curve per block
  Block switch   choices aligned to the silent block switches
  History        P(right) split by the previous trial and the current block
  Timeline       one session, trial by trial
  Timing         response / first-movement times vs contrast
  Block lengths  how long the biased blocks last
  Table          the filtered trials themselves (click a header to sort)

Hover over any point for its numbers. In plots: scroll to zoom, drag to pan,
right-click -> "View All" to reset. All numbers come from ibl_analysis.py,
the same code explore_ibl_data.ipynb uses.
"""

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

# A windowed build (PyInstaller, console=False) has no console, so
# sys.stdout/stderr are None and anything that prints -- tracebacks, ONE's
# download progress bars, logging set up at import time -- would crash.
# Send it to a log file instead. Must run before importing ONE.
LOG_PATH = Path(tempfile.gettempdir()) / "ibl_explorer.log"
if sys.stdout is None or sys.stderr is None:
    _log = open(LOG_PATH, "w", buffering=1, encoding="utf-8")
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import fetch_ibl_data as fetch
import ibl_analysis as ia

FROZEN = getattr(sys, "frozen", False)  # running as the PyInstaller-built app
HERE = Path(__file__).parent            # in the built app: its bundled-files folder
DEFAULT_CSV = HERE / "data" / "ibl_trials.csv"
# Open/Save dialogs start in the repo's data folder, or the home folder for the app
DIALOG_DIR = Path.home() if FROZEN else DEFAULT_CSV.parent
C = ia.COLORS
CONTRAST_LEVELS = [0.0, 0.0625, 0.125, 0.25, 1.0]
MAX_CONTRAST_CHOICES = [
    ("0 % only", 0.0),
    ("≤ 6.25 %", 0.0625),
    ("≤ 12.5 %", 0.125),
    ("≤ 25 %", 0.25),
    ("All contrasts", None),
]
LOG_TICKS = [(v, f"{10 ** v:g}") for v in (-2, -1, 0, 1)]  # log10(seconds) -> label
FINE_LOG_TICKS = [(np.log10(v), f"{v:g}") for v in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10)]
TABLE_COLUMNS = [
    "session_idx", "date", "trial_in_session", "signed_contrast", "choice",
    "chose_right", "correct", "probabilityLeft", "trial_in_block", "block_len",
    "response_time", "movement_time",
]

pg.setConfigOptions(background=C["surface"], foreground=C["ink_secondary"], antialias=True)


# =============================================================================
# Small plotting helpers
# =============================================================================

def mkpen(color, width=2, style=None):
    pen = pg.mkPen(color=color, width=width)
    if style == "dash":
        pen.setStyle(QtCore.Qt.DashLine)
    elif style == "dot":
        pen.setStyle(QtCore.Qt.DotLine)
    return pen


def tint(color, alpha):
    c = QtGui.QColor(color)
    c.setAlpha(alpha)
    return c


def swatch(color):
    pixmap = QtGui.QPixmap(12, 12)
    pixmap.fill(QtGui.QColor(color))
    return QtGui.QIcon(pixmap)


def style_plot(plot, title, xlabel, ylabel):
    plot.setTitle(title, color=C["ink"], size="11pt")
    plot.setLabel("bottom", xlabel)
    plot.setLabel("left", ylabel)
    plot.showGrid(x=True, y=True, alpha=0.15)
    for side in ("bottom", "left"):
        plot.getAxis(side).enableAutoSIPrefix(False)
        plot.getAxis(side).setPen(mkpen(C["axis"], 1))
        plot.getAxis(side).setTextPen(C["ink_secondary"])


def add_line(plot, x, y, pen, name=None, **kwargs):
    """plot.plot with numpy inputs (pyqtgraph indexes obj[0], which breaks
    on a filtered pandas Series whose index doesn't start at 0)."""
    return plot.plot(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                     pen=pen, name=name, **kwargs)


def add_points(plot, x, y, color, tips, filled=True, size=9, symbol="o", invisible=False):
    """Scatter points with a hover tooltip per point. Filled points get a
    surface-colored ring so overlapping marks stay separable; hollow points
    (used for errors / secondary series) are outlined in the series color.
    invisible=True makes hover-only targets, e.g. on top of bars."""
    if invisible:
        pen, brush = pg.mkPen(None), pg.mkBrush(0, 0, 0, 0)
    else:
        pen = mkpen(C["surface"] if filled else color, 1.5)
        brush = pg.mkBrush(color if filled else C["surface"])
    item = pg.ScatterPlotItem(
        x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=float),
        size=size, symbol=symbol, pen=pen, brush=brush,
        data=list(tips), hoverable=True, hoverSize=size + 4,
        hoverPen=mkpen(C["ink"], 1.5),
        tip=lambda x, y, data: data,
    )
    plot.addItem(item)
    return item


def add_errorbars(plot, x, y, sem, color):
    plot.addItem(pg.ErrorBarItem(x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=float),
                                 height=2 * np.asarray(sem, dtype=float), beam=0,
                                 pen=mkpen(color, 1.5)))


def add_reference_line(plot, value, angle=0, label=None):
    line = pg.InfiniteLine(pos=value, angle=angle, pen=mkpen(C["muted"], 1, "dot"),
                           label=label, labelOpts={"color": C["muted"], "position": 0.95})
    plot.addItem(line)


def show_message(plot, text):
    item = pg.TextItem(text, color=C["muted"], anchor=(0.5, 0.5))
    plot.addItem(item)
    item.setPos(0, 0)


def pct(contrast):
    return f"{100 * contrast:g} %"


# =============================================================================
# Background jobs (database access must not freeze the window)
# =============================================================================

class Job(QtCore.QThread):
    """Run fn(report, *args) in a worker thread. `report(i, n, text)`
    updates the progress bar from inside fn."""

    progress = QtCore.Signal(int, int, str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    def run(self):
        try:
            self.done.emit(self.fn(self.progress.emit, *self.args))
        except Exception as exc:  # shown in the GUI, not raised
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DataFrameModel(QtCore.QAbstractTableModel):
    """Read-only Qt view of a DataFrame (numeric sort via UserRole)."""

    def __init__(self, df=None):
        super().__init__()
        self.df = pd.DataFrame() if df is None else df

    def set_frame(self, df):
        self.beginResetModel()
        self.df = df
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.df)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return self.df.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        value = self.df.iat[index.row(), index.column()]
        if role == QtCore.Qt.UserRole:
            return value.item() if hasattr(value, "item") else value
        if role != QtCore.Qt.DisplayRole:
            return None
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:.4g}"
        return str(value)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        return str(self.df.columns[section]) if orientation == QtCore.Qt.Horizontal else str(section + 1)


# =============================================================================
# Main window
# =============================================================================

class Explorer(QtWidgets.QMainWindow):
    def __init__(self, csv_path=None):
        super().__init__()
        self.setWindowTitle("IBL trial explorer")
        self.resize(1440, 900)

        self.raw = None        # trials as loaded (what "Save CSV" writes)
        self.data = None       # ia.prepare(raw): all derived columns
        self.view = None       # filtered rows of `data`
        self.one = None        # ONE connection, opened on first download
        self.found = []        # [(start_time, eid)] from "Find sessions"
        self.job = None
        self.dirty = set()

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self._build_side_panel())
        self.tabs = QtWidgets.QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 1110])
        self.setCentralWidget(splitter)

        self.draw_fns = []
        for name, build, draw in [
            ("Sessions", self._build_sessions_tab, self.draw_sessions),
            ("Psychometric", self._build_psychometric_tab, self.draw_psychometric),
            ("Block switch", self._build_switch_tab, self.draw_switch),
            ("History", self._build_history_tab, self.draw_history),
            ("Timeline", self._build_timeline_tab, self.draw_timeline),
            ("Timing", self._build_timing_tab, self.draw_timing),
            ("Block lengths", self._build_lengths_tab, self.draw_lengths),
            ("Table", self._build_table_tab, self.draw_table),
        ]:
            self.tabs.addTab(build(), name)
            self.draw_fns.append(draw)
        self.tabs.currentChanged.connect(lambda _: self.draw_current())

        # Checking "All" toggles every session item; redraw once, not 40 times.
        self.refresh_timer = QtCore.QTimer(singleShot=True, interval=150)
        self.refresh_timer.timeout.connect(self.apply_filters)

        if csv_path is not None and Path(csv_path).exists():
            self.load_csv(Path(csv_path))
        else:
            self.statusBar().showMessage("No data loaded: open a CSV or download a subject.")

    # ------------------------------------------------------------------ panel
    def _build_side_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # --- Data source
        data_box = QtWidgets.QGroupBox("Data")
        data_layout = QtWidgets.QVBoxLayout(data_box)
        self.source_label = QtWidgets.QLabel("No data loaded")
        self.source_label.setWordWrap(True)
        data_layout.addWidget(self.source_label)
        row = QtWidgets.QHBoxLayout()
        open_btn = QtWidgets.QPushButton("Open CSV...")
        open_btn.clicked.connect(self.open_csv_dialog)
        self.save_btn = QtWidgets.QPushButton("Save CSV...")
        self.save_btn.clicked.connect(self.save_csv_dialog)
        self.save_btn.setEnabled(False)
        row.addWidget(open_btn)
        row.addWidget(self.save_btn)
        data_layout.addLayout(row)
        layout.addWidget(data_box)

        # --- Download from OpenAlyx
        alyx_box = QtWidgets.QGroupBox("Download from the IBL database")
        form = QtWidgets.QFormLayout(alyx_box)
        self.subject_edit = QtWidgets.QLineEdit(fetch.SUBJECT)
        self.project_edit = QtWidgets.QLineEdit(fetch.PROJECT)
        self.project_edit.setPlaceholderText("any project")
        form.addRow("Subject", self.subject_edit)
        form.addRow("Project", self.project_edit)
        self.find_btn = QtWidgets.QPushButton("Find sessions")
        self.find_btn.clicked.connect(self.find_sessions)
        form.addRow(self.find_btn)
        self.found_label = QtWidgets.QLabel(f"Task protocol: {fetch.TASK_PROTOCOL}")
        self.found_label.setWordWrap(True)
        form.addRow(self.found_label)
        range_row = QtWidgets.QHBoxLayout()
        self.from_spin = QtWidgets.QSpinBox()
        self.to_spin = QtWidgets.QSpinBox()
        for spin in (self.from_spin, self.to_spin):
            spin.setRange(1, 1)
            spin.setEnabled(False)
        range_row.addWidget(self.from_spin)
        range_row.addWidget(QtWidgets.QLabel("to"))
        range_row.addWidget(self.to_spin)
        form.addRow("Sessions", range_row)
        self.download_btn = QtWidgets.QPushButton("Download && load")
        self.download_btn.clicked.connect(self.download_sessions)
        self.download_btn.setEnabled(False)
        form.addRow(self.download_btn)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.hide()
        form.addRow(self.progress)
        layout.addWidget(alyx_box)

        # --- Filters
        filt_box = QtWidgets.QGroupBox("Filter")
        filt = QtWidgets.QVBoxLayout(filt_box)
        filt.addWidget(QtWidgets.QLabel("Sessions"))
        self.session_list = QtWidgets.QListWidget()
        self.session_list.itemChanged.connect(lambda _: self.refresh_timer.start())
        filt.addWidget(self.session_list, stretch=1)
        row = QtWidgets.QHBoxLayout()
        for text, state in (("All", True), ("None", False)):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(lambda _, s=state: self.check_all_sessions(s))
            row.addWidget(btn)
        filt.addLayout(row)

        self.block_checks = {}
        for block in (ia.LEFT_BLOCK, ia.NEUTRAL_BLOCK, ia.RIGHT_BLOCK):
            box = QtWidgets.QCheckBox(ia.BLOCK_LABELS[block])
            box.setIcon(swatch(ia.BLOCK_COLORS[block]))
            box.setChecked(True)
            box.toggled.connect(lambda _: self.refresh_timer.start())
            self.block_checks[block] = box
            filt.addWidget(box)

        skip_row = QtWidgets.QHBoxLayout()
        skip_row.addWidget(QtWidgets.QLabel("Skip each session's first"))
        self.skip_spin = QtWidgets.QSpinBox()
        self.skip_spin.setRange(0, 1000)
        self.skip_spin.setSuffix(" trials")
        self.skip_spin.setToolTip("90 = the unbiased block every session opens with")
        self.skip_spin.valueChanged.connect(lambda _: self.refresh_timer.start())
        skip_row.addWidget(self.skip_spin)
        filt.addLayout(skip_row)

        self.count_label = QtWidgets.QLabel("")
        filt.addWidget(self.count_label)
        layout.addWidget(filt_box, stretch=1)
        return panel

    @staticmethod
    def _controls(*widgets, hint=None):
        """A row of tab controls (filters sit above the chart)."""
        row = QtWidgets.QHBoxLayout()
        for w in widgets:
            if isinstance(w, str):
                row.addWidget(QtWidgets.QLabel(w))
            else:
                row.addWidget(w)
        row.addStretch(1)
        box = QtWidgets.QVBoxLayout()
        box.addLayout(row)
        if hint:
            label = QtWidgets.QLabel(hint)
            label.setWordWrap(True)
            label.setStyleSheet(f"color: {C['ink_secondary']};")
            box.addWidget(label)
        return box

    def _tab(self, controls, plots):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        if controls is not None:
            layout.addLayout(controls)
        layout.addWidget(plots, stretch=1)
        return page

    # ------------------------------------------------------------- tab builds
    def _build_sessions_tab(self):
        self.sess_glw = pg.GraphicsLayoutWidget()
        self.sess_plots = [self.sess_glw.addPlot(row=i, col=0) for i in range(3)]
        for p in self.sess_plots[1:]:
            p.setXLink(self.sess_plots[0])
        hint = ("Block-following = on 0 % trials in a biased block, how often the mouse "
                "chose the side that block favors. 0.5 means it ignores the block.")
        return self._tab(self._controls(hint=hint), self.sess_glw)

    def _build_psychometric_tab(self):
        self.psy_even = QtWidgets.QCheckBox("Evenly spaced contrasts")
        self.psy_even.setChecked(True)
        self.psy_even.toggled.connect(lambda _: self.redraw(self.draw_psychometric))
        self.psy_plot = pg.PlotWidget()
        hint = ("If the mouse uses the hidden block, the curves separate, most of all "
                "at 0 % where the stimulus says nothing.")
        return self._tab(self._controls(self.psy_even, hint=hint), self.psy_plot)

    def _build_switch_tab(self):
        self.sw_before = QtWidgets.QSpinBox()
        self.sw_before.setRange(1, 60)
        self.sw_before.setValue(10)
        self.sw_after = QtWidgets.QSpinBox()
        self.sw_after.setRange(1, 100)
        self.sw_after.setValue(40)
        self.sw_contrast = QtWidgets.QComboBox()
        for label, _ in MAX_CONTRAST_CHOICES:
            self.sw_contrast.addItem(label)
        self.sw_split = QtWidgets.QCheckBox("Split by new block")
        self.sw_split.setChecked(True)
        for w in (self.sw_before, self.sw_after):
            w.valueChanged.connect(lambda _: self.redraw(self.draw_switch))
        self.sw_contrast.currentIndexChanged.connect(lambda _: self.redraw(self.draw_switch))
        self.sw_split.toggled.connect(lambda _: self.redraw(self.draw_switch))
        self.sw_plot = pg.PlotWidget()
        hint = ("Use low contrasts. With visible stimuli the curve mostly shows that "
                "stimuli start coming from the new side, not that the mouse adapted.")
        controls = self._controls("Trials before", self.sw_before, "after", self.sw_after,
                                  "Contrast", self.sw_contrast, self.sw_split, hint=hint)
        return self._tab(controls, self.sw_plot)

    def _build_history_tab(self):
        self.hist_contrast = QtWidgets.QComboBox()
        for c in CONTRAST_LEVELS:
            self.hist_contrast.addItem(pct(c), c)
        self.hist_contrast.currentIndexChanged.connect(lambda _: self.redraw(self.draw_history))
        self.hist_plot = pg.PlotWidget()
        hint = ("Within one block color, differences between the groups show the effect "
                "of the previous trial; the gap between colors is the block's effect.")
        return self._tab(self._controls("Current trial contrast", self.hist_contrast, hint=hint),
                         self.hist_plot)

    def _build_timeline_tab(self):
        self.tl_session = QtWidgets.QComboBox()
        self.tl_session.currentIndexChanged.connect(lambda _: self.redraw(self.draw_timeline))
        self.tl_window = QtWidgets.QSpinBox()
        self.tl_window.setRange(1, 100)
        self.tl_window.setValue(10)
        self.tl_window.valueChanged.connect(lambda _: self.redraw(self.draw_timeline))
        self.tl_glw = pg.GraphicsLayoutWidget()
        self.tl_plots = [self.tl_glw.addPlot(row=i, col=0) for i in range(3)]
        for p in self.tl_plots[1:]:
            p.setXLink(self.tl_plots[0])
        self.tl_glw.ci.layout.setRowStretchFactor(1, 2)
        hint = ("Shows the whole session (the block and skip filters don't apply here). "
                "Filled marks = rewarded, hollow = error.")
        controls = self._controls("Session", self.tl_session, "Running average over",
                                  self.tl_window, "trials", hint=hint)
        return self._tab(controls, self.tl_glw)

    def _build_timing_tab(self):
        self.tm_measure = QtWidgets.QComboBox()
        for key, label in ia.RT_MEASURES.items():
            self.tm_measure.addItem(label, key)
        self.tm_measure.currentIndexChanged.connect(lambda _: self.redraw(self.draw_timing))
        self.tm_glw = pg.GraphicsLayoutWidget()
        self.tm_plots = [self.tm_glw.addPlot(row=0, col=i) for i in range(2)]
        hint = ("First-movement time is often ~0 or negative in these sessions: mice keep "
                "nudging the wheel through the go cue. Response time (wheel reached the "
                "threshold) is the cleaner measure here.")
        return self._tab(self._controls("Measure", self.tm_measure, hint=hint), self.tm_glw)

    def _build_lengths_tab(self):
        self.len_bin = QtWidgets.QSpinBox()
        self.len_bin.setRange(1, 20)
        self.len_bin.setValue(5)
        self.len_bin.setSuffix(" trials")
        self.len_bin.valueChanged.connect(lambda _: self.redraw(self.draw_lengths))
        self.len_plot = pg.PlotWidget()
        hint = ("Each session's last block is left out (the session ended it early). "
                "Compare with DNN-window's 5-trial memory in rnn_vs_dnn_ibl.py.")
        return self._tab(self._controls("Bin width", self.len_bin, hint=hint), self.len_plot)

    def _build_table_tab(self):
        self.table_model = DataFrameModel()
        proxy = QtCore.QSortFilterProxyModel()
        proxy.setSourceModel(self.table_model)
        proxy.setSortRole(QtCore.Qt.UserRole)
        self.table = QtWidgets.QTableView()
        self.table.setModel(proxy)
        self.table.horizontalHeader().setSortIndicator(-1, QtCore.Qt.AscendingOrder)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setDefaultSectionSize(20)
        return self._tab(None, self.table)

    # ---------------------------------------------------------- data loading
    def open_csv_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open trials CSV", str(DIALOG_DIR), "CSV files (*.csv)")
        if path:
            self.load_csv(Path(path))

    def save_csv_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save trials CSV", str(DIALOG_DIR / "trials.csv"), "CSV files (*.csv)")
        if path:
            self.raw.to_csv(path, index=False)
            self.statusBar().showMessage(f"Saved {len(self.raw):,} trials to {path}")

    def load_csv(self, path):
        self.set_data(pd.read_csv(path), f"{path.name}")

    def set_data(self, raw, source):
        self.raw = raw
        self.data = ia.prepare(raw)
        sessions = self.data.groupby("session_idx").agg(date=("date", "first"), n=("choice", "size"))
        self.session_list.blockSignals(True)
        self.session_list.clear()
        for idx, row in sessions.iterrows():
            item = QtWidgets.QListWidgetItem(f"{idx + 1:>3}   {row.date}   {row.n} trials")
            item.setData(QtCore.Qt.UserRole, int(idx))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.session_list.addItem(item)
        self.session_list.blockSignals(False)
        has_timing = "response_time" in self.data
        self.tabs.setTabEnabled(5, has_timing)
        self.source_label.setText(
            f"<b>{source}</b><br>{len(self.data):,} trials, {len(sessions)} sessions "
            f"({sessions.date.iloc[0]} to {sessions.date.iloc[-1]})"
            + ("" if has_timing else "<br>No timing columns: Timing tab disabled."))
        self.save_btn.setEnabled(True)
        self.apply_filters()

    def check_all_sessions(self, checked):
        state = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
        self.session_list.blockSignals(True)
        for i in range(self.session_list.count()):
            self.session_list.item(i).setCheckState(state)
        self.session_list.blockSignals(False)
        self.apply_filters()

    def apply_filters(self):
        if self.data is None:
            return
        sessions = [self.session_list.item(i).data(QtCore.Qt.UserRole)
                    for i in range(self.session_list.count())
                    if self.session_list.item(i).checkState() == QtCore.Qt.Checked]
        blocks = [b for b, box in self.block_checks.items() if box.isChecked()]
        self.view = ia.filter_trials(self.data, sessions, blocks, self.skip_spin.value())
        self.count_label.setText(f"Showing {len(self.view):,} trials from "
                                 f"{self.view['session_idx'].nunique()} sessions")

        # Timeline lists the checked sessions; keep the current pick if possible.
        current = self.tl_session.currentData()
        self.tl_session.blockSignals(True)
        self.tl_session.clear()
        for idx in sessions:
            date = self.data.loc[self.data["session_idx"] == idx, "date"].iloc[0]
            self.tl_session.addItem(f"{idx + 1}  ({date})", idx)
        if current in sessions:
            self.tl_session.setCurrentIndex(sessions.index(current))
        self.tl_session.blockSignals(False)

        self.dirty = set(range(self.tabs.count()))
        self.draw_current()

    def draw_current(self):
        i = self.tabs.currentIndex()
        if self.view is not None and i in self.dirty:
            self.dirty.discard(i)
            self.draw_fns[i]()

    def redraw(self, fn):
        if self.view is not None:
            fn()

    # ------------------------------------------------------ OpenAlyx access
    def _connect(self):
        if self.one is None:
            self.one = fetch.connect()
        return self.one

    def _start_job(self, job, on_done):
        self.job = job
        for w in (self.find_btn, self.download_btn):
            w.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        job.progress.connect(self._on_progress)
        job.done.connect(on_done)
        job.failed.connect(self._on_failed)
        job.finished.connect(self._on_job_finished)
        job.start()

    def _on_progress(self, i, n, text):
        if n:
            self.progress.setRange(0, n)
            self.progress.setValue(i)
        self.statusBar().showMessage(text)

    def _on_failed(self, message):
        self.statusBar().showMessage("Database request failed")
        QtWidgets.QMessageBox.warning(self, "Database request failed", message)

    def _on_job_finished(self):
        self.progress.hide()
        self.find_btn.setEnabled(True)
        self.download_btn.setEnabled(bool(self.found))

    def find_sessions(self):
        subject = self.subject_edit.text().strip()
        project = self.project_edit.text().strip() or None

        def work(report):
            report(0, 0, "Connecting to OpenAlyx...")
            one = self._connect()
            report(0, 0, f"Searching sessions of {subject}...")
            return fetch.find_sessions(one, subject, project=project)

        self._start_job(Job(work), self._on_found)

    def _on_found(self, sessions):
        self.found = sessions
        n = len(sessions)
        if n == 0:
            self.found_label.setText("No sessions found. Check the subject and project.")
        else:
            self.found_label.setText(f"{n} sessions, {sessions[0][0][:10]} to {sessions[-1][0][:10]}")
        for spin in (self.from_spin, self.to_spin):
            spin.setRange(1, max(n, 1))
            spin.setEnabled(n > 0)
        self.from_spin.setValue(1)
        self.to_spin.setValue(min(n, fetch.MAX_SESSIONS))
        self.statusBar().showMessage(f"Found {n} sessions")

    def download_sessions(self):
        first, last = self.from_spin.value(), self.to_spin.value()
        if last < first:
            first, last = last, first
        picked = [(i, *self.found[i]) for i in range(first - 1, last)]
        subject = self.subject_edit.text().strip()

        def work(report):
            one = self._connect()
            frames = []
            for k, (idx, start, eid) in enumerate(picked):
                report(k, len(picked), f"Downloading session {idx + 1} ({start[:10]}), "
                                       f"{k + 1} of {len(picked)}")
                trials, _ = fetch.load_session_trials(one, eid, idx, start)
                frames.append(trials)
            report(len(picked), len(picked), "Download finished")
            return pd.concat(frames, ignore_index=True)

        self._start_job(Job(work), lambda df: self.set_data(
            df, f"{subject}, sessions {first} to {last} (OpenAlyx, not saved)"))

    # ------------------------------------------------------------ the tabs
    def draw_sessions(self):
        p_count, p_acc, p_follow = self.sess_plots
        for p in self.sess_plots:
            p.clear()
        if p_acc.legend is None:
            p_acc.addLegend(offset=(-10, -10))
        p_acc.legend.clear()
        s = ia.session_summary(self.view)
        if s.empty:
            show_message(p_count, "No trials match the filters")
            return
        x = s["session_idx"].to_numpy() + 1
        tips = [f"Session {i}  ({d})\n{n} trials\naccuracy {a:.1%}\neasy-trial accuracy {e:.1%}\n"
                f"follows block on 0 %: {f:.1%}"
                for i, d, n, a, e, f in zip(x, s["date"], s["n_trials"], s["accuracy"],
                                            s["easy_accuracy"], s["follows_block_0"])]

        style_plot(p_count, "Trials per session", "", "trials")
        p_count.addItem(pg.BarGraphItem(x=x, height=s["n_trials"].to_numpy(), width=0.8,
                                        brush=C["ink_secondary"], pen=mkpen(C["surface"], 1)))
        add_points(p_count, x, s["n_trials"], C["ink_secondary"], tips, size=14, invisible=True)

        style_plot(p_acc, "Accuracy", "", "fraction correct")
        add_line(p_acc, x, s["accuracy"], pen=mkpen(C["ink"]), name="All trials")
        add_points(p_acc, x, s["accuracy"], C["ink"], tips)
        add_line(p_acc, x, s["easy_accuracy"], pen=mkpen(C["ink_secondary"], 2, "dash"),
                   name=f"Easy trials ({pct(ia.EASY_CONTRAST)})")
        add_points(p_acc, x, s["easy_accuracy"], C["ink_secondary"], tips, filled=False)
        p_acc.setYRange(0.5, 1.0)

        style_plot(p_follow, "Block-following on 0 % trials", "session", "P(choice = block side)")
        add_reference_line(p_follow, 0.5, label="ignores block")
        add_line(p_follow, x, s["follows_block_0"], pen=mkpen(C["ink"]))
        add_points(p_follow, x, s["follows_block_0"], C["ink"], tips)
        follow = s["follows_block_0"].dropna()
        if len(follow):
            p_follow.setYRange(min(0.45, follow.min()), max(0.55, follow.max()))

    def draw_psychometric(self):
        p = self.psy_plot
        p.clear()
        if p.plotItem.legend is None:
            p.addLegend(offset=(10, 10))
        p.plotItem.legend.clear()
        style_plot(p.plotItem, "Psychometric curves by block", "signed contrast (right +)",
                   "P(chose right)")
        table = ia.psychometric(self.view)
        levels = sorted(table["signed_contrast"].unique())
        even = self.psy_even.isChecked()
        to_x = {c: (i if even else 100 * c) for i, c in enumerate(levels)}
        axis = p.getAxis("bottom")
        axis.setTicks([[(to_x[c], f"{100 * c:g}") for c in levels]] if even else None)
        add_reference_line(p, 0.5)
        if table.empty:
            show_message(p, "No trials match the filters")
            return
        add_reference_line(p, to_x[0.0] if 0.0 in to_x else 0, angle=90)
        for block in (ia.LEFT_BLOCK, ia.NEUTRAL_BLOCK, ia.RIGHT_BLOCK):
            t = table[table["probabilityLeft"] == block]
            if t.empty:
                continue
            color = ia.BLOCK_COLORS[block]
            x = t["signed_contrast"].map(to_x)
            add_line(p, x, t["p_right"], pen=mkpen(color), name=ia.BLOCK_LABELS[block])
            add_errorbars(p, x, t["p_right"], t["sem"], color)
            tips = [f"{ia.BLOCK_LABELS[block]}\ncontrast {100 * c:+g} %\n"
                    f"P(right) = {v:.2f} +/- {e:.2f}\nn = {n}"
                    for c, v, e, n in zip(t["signed_contrast"], t["p_right"], t["sem"], t["n"])]
            add_points(p, x, t["p_right"], color, tips)
        p.setYRange(0, 1)

    def draw_switch(self):
        p = self.sw_plot
        p.clear()
        if p.plotItem.legend is None:
            p.addLegend(offset=(-10, -10))
        p.plotItem.legend.clear()
        label, max_c = MAX_CONTRAST_CHOICES[self.sw_contrast.currentIndex()]
        split = self.sw_split.isChecked()
        curve = ia.block_switch_curve(self.view, self.sw_before.value(), self.sw_after.value(),
                                      max_contrast=max_c, split=split)
        style_plot(p.plotItem, f"Choices around block switches ({label})",
                   "trials since switch (0 = first trial of new block)",
                   "P(choice = new block's side)")
        add_reference_line(p, 0.5)
        add_reference_line(p, -0.5, angle=90, label="switch")
        if curve.empty:
            show_message(p, "No 0.2 <-> 0.8 switches in the filtered trials")
            return
        groups = ([(b, curve[curve["new_block"] == b]) for b in (ia.LEFT_BLOCK, ia.RIGHT_BLOCK)]
                  if split else [(None, curve)])
        for block, g in groups:
            if g.empty:
                continue
            color = ia.BLOCK_COLORS[block] if block else C["ink"]
            name = ("Into " + ia.BLOCK_LABELS[block][0].lower() + ia.BLOCK_LABELS[block][1:]
                    if block else "All switches")
            x, y, e = g["offset"].to_numpy(), g["p"].to_numpy(), g["sem"].to_numpy()
            hi = add_line(p, x, y + e, pen=None)
            lo = add_line(p, x, y - e, pen=None)
            p.addItem(pg.FillBetweenItem(hi, lo, brush=tint(color, 45)))
            add_line(p, x, y, pen=mkpen(color), name=name)
            tips = [f"{name}\n{o:+d} trials\nP = {v:.2f} +/- {s:.2f}\nn = {n}"
                    for o, v, s, n in zip(g["offset"], g["p"], g["sem"], g["n"])]
            add_points(p, x, y, color, tips, size=7)
        p.setYRange(0, 1)

    def draw_history(self):
        p = self.hist_plot
        p.clear()
        if p.plotItem.legend is None:
            p.addLegend(offset=(10, 10))
        p.plotItem.legend.clear()
        contrast = self.hist_contrast.currentData()
        table = ia.history_table(self.view, contrast=contrast)
        style_plot(p.plotItem, f"P(chose right) at {pct(contrast)} contrast, by previous trial",
                   "previous trial", "P(chose right)")
        p.getAxis("bottom").setTicks([list(enumerate(ia.HISTORY_ORDER))])
        add_reference_line(p, 0.5)
        if table.empty:
            show_message(p, "No trials match the filters")
            return
        blocks = [b for b in (ia.LEFT_BLOCK, ia.NEUTRAL_BLOCK, ia.RIGHT_BLOCK)
                  if b in set(table["probabilityLeft"])]
        width = 0.8 / len(blocks)
        for k, block in enumerate(blocks):
            t = table[table["probabilityLeft"] == block]
            x = t["previous"].cat.codes.to_numpy() - 0.4 + width * (k + 0.5)
            color = ia.BLOCK_COLORS[block]
            bars = pg.BarGraphItem(x=x, height=t["p_right"].to_numpy(), width=width * 0.92,
                                   brush=color, pen=mkpen(C["surface"], 2))
            p.addItem(bars)
            p.plotItem.legend.addItem(bars, ia.BLOCK_LABELS[block])
            add_errorbars(p, x, t["p_right"], t["sem"], C["ink"])
            tips = [f"Previous: {prev}\n{ia.BLOCK_LABELS[block]}\nP(right) = {v:.2f} +/- {e:.2f}\n"
                    f"n = {n}" for prev, v, e, n in zip(t["previous"], t["p_right"], t["sem"], t["n"])]
            add_points(p, x, t["p_right"], color, tips, size=16, invisible=True)
        p.setYRange(0, 1)
        p.setXRange(-0.6, 3.6)

    def draw_timeline(self):
        p_stim, p_choice, p_rt = self.tl_plots
        for p in self.tl_plots:
            p.clear()
        idx = self.tl_session.currentData()
        if idx is None:
            show_message(p_choice, "No session selected")
            return
        s = self.data[self.data["session_idx"] == idx]
        t = s["trial_in_session"].to_numpy()
        side_color = np.where(s["stim_right"], C["right"], C["left"])
        tips = [f"Trial {i}\nstimulus {100 * c:+g} %\nchose {'right' if r else 'left'} "
                f"({'rewarded' if ok else 'error'})\nblock pLeft {b}\n"
                + (f"response time {rt:.2f} s" if rt == rt else "")
                for i, c, r, ok, b, rt in zip(t, s["signed_contrast"], s["chose_right"],
                                              s["correct"], s["probabilityLeft"],
                                              s.get("response_time", pd.Series(np.nan, index=s.index)))]

        style_plot(p_stim, f"Session {idx + 1} ({s['date'].iloc[0]}): stimulus", "",
                   "contrast % (right +)")
        for color in (C["left"], C["right"]):
            m = side_color == color
            add_points(p_stim, t[m], 100 * s["signed_contrast"].to_numpy()[m], color,
                       np.array(tips, dtype=object)[m], size=6)

        style_plot(p_choice, "Choices and the hidden block", "", "P(chose right)")
        for _, blk in s.groupby("block_id"):
            p_choice.addItem(pg.LinearRegionItem(
                (blk["trial_in_session"].iloc[0] - 0.5, blk["trial_in_session"].iloc[-1] + 0.5),
                movable=False, brush=tint(ia.BLOCK_COLORS[blk["probabilityLeft"].iloc[0]], 28),
                pen=mkpen(C["surface"], 0)))
        add_line(p_choice, t, 1 - s["probabilityLeft"], pen=mkpen(C["ink_secondary"], 1.5, "dash"))
        running = s["chose_right"].astype(float).rolling(self.tl_window.value(), center=True,
                                                         min_periods=1).mean()
        add_line(p_choice, t, running, pen=mkpen(C["ink"]))
        chose_right = s["chose_right"].to_numpy()
        y = np.where(chose_right, 1.08, -0.08)
        for right, color in ((False, C["left"]), (True, C["right"])):
            for ok in (True, False):
                m = (chose_right == right) & (s["correct"].to_numpy() == ok)
                add_points(p_choice, t[m], y[m], color, np.array(tips, dtype=object)[m],
                           filled=ok, size=6)
        p_choice.setYRange(-0.15, 1.15)
        p_choice.setLabel("bottom", "")

        style_plot(p_rt, "Response time", "trial in session", "seconds (log scale)")
        if "response_time" in s:
            ok = (s["response_time"] > 0).to_numpy()
            add_points(p_rt, t[ok], np.log10(s["response_time"].to_numpy()[ok]),
                       C["ink_secondary"], np.array(tips, dtype=object)[ok], size=5)
            p_rt.getAxis("left").setTicks([LOG_TICKS])
        else:
            show_message(p_rt, "No timing columns in this data")
        p_stim.setXRange(t.min() - 1, t.max() + 1)

    def draw_timing(self):
        p_chrono, p_hist = self.tm_plots
        for p in self.tm_plots:
            p.clear()
        if "response_time" not in self.view:
            return
        if p_chrono.legend is None:
            p_chrono.addLegend(offset=(10, -10))
        p_chrono.legend.clear()
        measure = self.tm_measure.currentData()
        label = ia.RT_MEASURES[measure]
        table = ia.chronometric(self.view, measure)

        style_plot(p_chrono, f"{label.split(' (')[0]}: median and quartiles", "|contrast|",
                   "seconds (log scale)")
        p_chrono.getAxis("left").setTicks([FINE_LOG_TICKS])
        p_chrono.getAxis("bottom").setTicks([[(i, pct(c)) for i, c in enumerate(CONTRAST_LEVELS)]])
        to_x = {c: i for i, c in enumerate(CONTRAST_LEVELS)}
        for correct, color, style, name in ((True, C["ink"], None, "Rewarded"),
                                            (False, C["ink_secondary"], "dash", "Error")):
            t = table[table["correct"] == correct]
            if t.empty:
                continue
            x = t["abs_contrast"].map(to_x).to_numpy(dtype=float) + (0.06 if not correct else -0.06)
            med, q25, q75 = (np.log10(t[k].to_numpy()) for k in ("median", "q25", "q75"))
            add_line(p_chrono, x, med, pen=mkpen(color, 2, style), name=name)
            p_chrono.addItem(pg.ErrorBarItem(x=x, y=med, top=q75 - med, bottom=med - q25,
                                             beam=0, pen=mkpen(color, 1.5)))
            tips = [f"{name}, |contrast| {pct(c)}\nmedian {m:.3f} s\nquartiles {a:.3f} to {b:.3f} s\n"
                    f"n = {n}" for c, m, a, b, n in zip(t["abs_contrast"], t["median"],
                                                         t["q25"], t["q75"], t["n"])]
            add_points(p_chrono, x, med, color, tips, filled=correct)

        values = self.view[measure]
        positive = values[values > 0]
        n_dropped = int(values.notna().sum() - len(positive))
        style_plot(p_hist, f"Distribution ({n_dropped:,} trials ≤ 0 s left out)",
                   "seconds (log scale)", "trials")
        if len(positive):
            edges = np.linspace(-2.5, 1.5, 81)
            counts, _ = np.histogram(np.log10(positive), bins=edges)
            add_line(p_hist, edges, counts, stepMode="center", pen=mkpen(C["ink"]),
                        fillLevel=0, brush=tint(C["ink"], 30))
            add_reference_line(p_hist, np.log10(positive.median()), angle=90,
                               label=f"median {positive.median():.3f} s")
            p_hist.getAxis("bottom").setTicks([LOG_TICKS])

    def draw_lengths(self):
        p = self.len_plot
        p.clear()
        if p.plotItem.legend is None:
            p.addLegend(offset=(-10, 10))
        p.plotItem.legend.clear()
        runs = ia.block_runs(self.view)
        style_plot(p.plotItem, f"Biased block lengths ({len(runs)} complete blocks)",
                   "block length (trials)", "blocks")
        if runs.empty:
            show_message(p, "No complete biased blocks in the filtered trials")
            return
        width = self.len_bin.value()
        edges = np.arange(0, runs["length"].max() + width + 1, width)
        for block in (ia.LEFT_BLOCK, ia.RIGHT_BLOCK):
            lengths = runs.loc[runs["probabilityLeft"] == block, "length"]
            if lengths.empty:
                continue
            counts, _ = np.histogram(lengths, bins=edges)
            add_line(p, edges, counts, stepMode="center", pen=mkpen(ia.BLOCK_COLORS[block]),
                   name=f"{ia.BLOCK_LABELS[block]}, median {lengths.median():g}")
        add_reference_line(p, runs["length"].median(), angle=90,
                           label=f"median {runs['length'].median():g}")

    def draw_table(self):
        columns = [c for c in TABLE_COLUMNS if c in self.view]
        self.table_model.set_frame(self.view[columns].reset_index(drop=True))
        self.table.resizeColumnsToContents()


def smoke_test(window, app, online):
    """Draw every tab (and optionally query OpenAlyx), then report.
    Used to check a fresh build; returns the process exit code."""
    errors = []
    sys.excepthook = lambda *exc: errors.append("".join(traceback.format_exception(*exc)))
    if window.data is None:
        errors.append(f"no data loaded from {DEFAULT_CSV}")
    else:
        for i in range(window.tabs.count()):
            window.tabs.setCurrentIndex(i)
            app.processEvents()
        print(f"Drew {window.tabs.count()} tabs from {len(window.data):,} trials")
    if online:
        try:
            sessions = fetch.find_sessions(fetch.connect(), fetch.SUBJECT)
            print(f"OpenAlyx: found {len(sessions)} sessions for {fetch.SUBJECT}")
        except Exception:
            errors.append(traceback.format_exc())
    print(f"Smoke test {'FAILED' if errors else 'passed'}")
    for e in errors:
        print(e)
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description="IBL trial explorer")
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV,
                        help="trials CSV (default: the bundled data/ibl_trials.csv)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="draw every tab, print a report and exit (for checking a build)")
    parser.add_argument("--online", action="store_true",
                        help="with --smoke-test: also query the IBL database")
    args = parser.parse_args()

    app = pg.mkQApp("IBL trial explorer")
    window = Explorer(args.csv)
    if args.smoke_test:
        sys.exit(smoke_test(window, app, args.online))

    # PyQt5 aborts on an exception escaping a slot; report it instead.
    def excepthook(*exc_info):
        traceback.print_exception(*exc_info)
        window.statusBar().showMessage(f"Error: {exc_info[1]}")
    sys.excepthook = excepthook

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
