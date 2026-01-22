import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@mock.patch('osprofiler.profiler.uuidutils.generate_uuid')
def test_profiler_get_id(self, mock_generate_uuid):
    mock_generate_uuid.return_value = '43'
    prof = profiler._Profiler('secret')
    prof.start('test')
    self.assertEqual(prof.get_id(), '43')