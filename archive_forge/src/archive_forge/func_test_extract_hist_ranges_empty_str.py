import io
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from traitlets.config.loader import Config
from IPython.core.history import HistoryAccessor, HistoryManager, extract_hist_ranges
def test_extract_hist_ranges_empty_str():
    instr = ''
    expected = [(0, 1, None)]
    actual = list(extract_hist_ranges(instr))
    assert actual == expected