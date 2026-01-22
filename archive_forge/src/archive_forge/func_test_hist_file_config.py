import io
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from traitlets.config.loader import Config
from IPython.core.history import HistoryAccessor, HistoryManager, extract_hist_ranges
def test_hist_file_config():
    cfg = Config()
    tfile = tempfile.NamedTemporaryFile(delete=False)
    cfg.HistoryManager.hist_file = Path(tfile.name)
    try:
        hm = HistoryManager(shell=get_ipython(), config=cfg)
        assert hm.hist_file == cfg.HistoryManager.hist_file
    finally:
        try:
            Path(tfile.name).unlink()
        except OSError:
            pass