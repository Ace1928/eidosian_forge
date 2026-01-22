import numpy as np
import scipy.sparse as ssp
import re
from unittest import mock
from nipype.pipeline.plugins.tools import report_crash
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
def test_report_crash():
    with mock.patch('pickle.dump', mock.MagicMock()) as mock_pickle_dump:
        with mock.patch('nipype.pipeline.plugins.tools.format_exception', mock.MagicMock()):
            mock_pickle_dump.return_value = True
            mock_node = mock.MagicMock(name='mock_node')
            mock_node._id = 'an_id'
            mock_node.config = {'execution': {'crashdump_dir': '.', 'crashfile_format': 'pklz'}}
            actual_crashfile = report_crash(mock_node)
            expected_crashfile = re.compile('.*/crash-.*-an_id-[0-9a-f\\-]*.pklz')
            assert expected_crashfile.match(actual_crashfile).group() == actual_crashfile
            assert mock_pickle_dump.call_count == 1