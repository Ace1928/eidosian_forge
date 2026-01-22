import io
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from traitlets.config.loader import Config
from IPython.core.history import HistoryAccessor, HistoryManager, extract_hist_ranges
def test_histmanager_disabled():
    """Ensure that disabling the history manager doesn't create a database."""
    cfg = Config()
    cfg.HistoryAccessor.enabled = False
    ip = get_ipython()
    with TemporaryDirectory() as tmpdir:
        hist_manager_ori = ip.history_manager
        hist_file = Path(tmpdir) / 'history.sqlite'
        cfg.HistoryManager.hist_file = hist_file
        try:
            ip.history_manager = HistoryManager(shell=ip, config=cfg)
            hist = ['a=1', 'def f():\n    test = 1\n    return test', "b='€Æ¾÷ß'"]
            for i, h in enumerate(hist, start=1):
                ip.history_manager.store_inputs(i, h)
            assert ip.history_manager.input_hist_raw == [''] + hist
            ip.history_manager.reset()
            ip.history_manager.end_session()
        finally:
            ip.history_manager = hist_manager_ori
    assert hist_file.exists() is False