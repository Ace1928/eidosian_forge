import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@mock.patch('osprofiler.profiler.uuidutils.generate_uuid')
def test_profiler_get_base_id_unset_case(self, mock_generate_uuid):
    mock_generate_uuid.return_value = '42'
    prof = profiler._Profiler('secret')
    self.assertEqual(prof.get_base_id(), '42')
    self.assertEqual(prof.get_parent_id(), '42')