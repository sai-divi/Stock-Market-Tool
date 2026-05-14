"""Build OverlayPredictor.exe with PyInstaller."""
import PyInstaller.__main__, sys
from pathlib import Path

ROOT = Path(__file__).parent
SEP = ";" if sys.platform == "win32" else ":"

PyInstaller.__main__.run([
    "--name=OverlayPredictor",
    "--console",
    "--onefile",
    "--clean",
    "--noconfirm",
    "--collect-all", "xgboost",
    f"{ROOT / 'app.py'}",
])
