import io
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from traitlets.config.loader import Config
from IPython.core.history import HistoryAccessor, HistoryManager, extract_hist_ranges
def test_get_tail_session_awareness():
    """Test .get_tail() is:
        - session specific in HistoryManager
        - session agnostic in HistoryAccessor
    same for .get_last_session_id()
    """
    ip = get_ipython()
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hist_file = tmp_path / 'history.sqlite'
        get_source = lambda x: x[2]
        hm1 = None
        hm2 = None
        ha = None
        try:
            hm1 = HistoryManager(shell=ip, hist_file=hist_file)
            hm1_last_sid = hm1.get_last_session_id
            ha = HistoryAccessor(hist_file=hist_file)
            ha_last_sid = ha.get_last_session_id
            hist1 = ['a=1', 'b=1', 'c=1']
            for i, h in enumerate(hist1 + [''], start=1):
                hm1.store_inputs(i, h)
            assert list(map(get_source, hm1.get_tail())) == hist1
            assert list(map(get_source, ha.get_tail())) == hist1
            sid1 = hm1_last_sid()
            assert sid1 is not None
            assert ha_last_sid() == sid1
            hm2 = HistoryManager(shell=ip, hist_file=hist_file)
            hm2_last_sid = hm2.get_last_session_id
            hist2 = ['a=2', 'b=2', 'c=2']
            for i, h in enumerate(hist2 + [''], start=1):
                hm2.store_inputs(i, h)
            tail = hm2.get_tail(n=3)
            assert list(map(get_source, tail)) == hist2
            tail = ha.get_tail(n=3)
            assert list(map(get_source, tail)) == hist2
            sid2 = hm2_last_sid()
            assert sid2 is not None
            assert ha_last_sid() == sid2
            assert sid2 != sid1
            assert hm1_last_sid() == sid1
            tail = hm1.get_tail(n=3)
            assert list(map(get_source, tail)) == hist1
            hist3 = ['a=3', 'b=3', 'c=3']
            for i, h in enumerate(hist3 + [''], start=5):
                hm1.store_inputs(i, h)
            tail = hm1.get_tail(n=7)
            assert list(map(get_source, tail)) == hist1 + [''] + hist3
            tail = hm2.get_tail(n=3)
            assert list(map(get_source, tail)) == hist2
            tail = ha.get_tail(n=3)
            assert list(map(get_source, tail)) == hist2
            assert hm1_last_sid() == sid1
            assert hm2_last_sid() == sid2
            assert ha_last_sid() == sid2
        finally:
            if hm1:
                hm1.save_thread.stop()
                hm1.db.close()
            if hm2:
                hm2.save_thread.stop()
                hm2.db.close()
            if ha:
                ha.db.close()