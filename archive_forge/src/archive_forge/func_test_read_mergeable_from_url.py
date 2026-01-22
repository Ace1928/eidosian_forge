from io import BytesIO
import breezy.bzr.bzrdir
import breezy.mergeable
import breezy.transport
import breezy.urlutils
from ... import errors, tests
from ...tests.per_transport import transport_test_permutations
from ...tests.scenarios import load_tests_apply_scenarios
from ...tests.test_transport import TestTransportImplementation
from ..bundle.serializer import write_bundle
def test_read_mergeable_from_url(self):
    info = self.read_mergeable_from_url(str(self.get_url(self.bundle_name)))
    revision = info.real_revisions[-1]
    self.assertEqual(b'commit-1', revision.revision_id)