from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def find_get_record_stream(self, calls, expected_count=1):
    """In a list of calls, find the last 'get_record_stream'.

        :param expected_count: The number of calls we should exepect to find.
            If a different number is found, an assertion is raised.
        """
    get_record_call = None
    call_count = 0
    for call in calls:
        if call[0] == 'get_record_stream':
            call_count += 1
            get_record_call = call
    self.assertEqual(expected_count, call_count)
    return get_record_call