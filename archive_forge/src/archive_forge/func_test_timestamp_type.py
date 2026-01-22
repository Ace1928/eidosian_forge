import io
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from traitlets.config.loader import Config
from IPython.core.history import HistoryAccessor, HistoryManager, extract_hist_ranges
def test_timestamp_type():
    ip = get_ipython()
    info = ip.history_manager.get_session_info()
    assert isinstance(info[1], datetime)