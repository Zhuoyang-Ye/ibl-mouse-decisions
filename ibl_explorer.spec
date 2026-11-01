# PyInstaller recipe for the desktop GUI as a standalone app (no Python needed).
#
#   pip install -r requirements-ibl.txt pyinstaller
#   pyinstaller --noconfirm ibl_explorer.spec
#
# Output: dist/IBL Explorer/ -- on Windows, "IBL Explorer.exe" plus an
# _internal folder; zip the whole folder to share it. GitHub Actions builds
# the Windows version (.github/workflows/windows-app.yml).

a = Analysis(
    ["ibl_explorer.py"],
    datas=[("data/ibl_trials.csv", "data")],  # the 40-session CSV opens on start
    # Packages that get pulled in through optional imports but never load
    # in the GUI (numba/llvmlite alone are ~120 MB). boto3 must stay: ONE
    # downloads from AWS S3 with it.
    excludes=["numba", "llvmlite", "scipy", "matplotlib", "IPython", "ipykernel",
              "jupyter_client", "zmq", "tornado", "torch", "tkinter", "pytest"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="IBL Explorer",
    console=False,  # a windowed app: no terminal window behind it
)
coll = COLLECT(exe, a.binaries, a.datas, name="IBL Explorer")
