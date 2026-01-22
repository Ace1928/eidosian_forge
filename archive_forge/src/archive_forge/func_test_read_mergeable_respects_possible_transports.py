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
def test_read_mergeable_respects_possible_transports(self):
    if not isinstance(self.get_transport(self.bundle_name), breezy.transport.ConnectedTransport):
        raise tests.TestSkipped('Need a ConnectedTransport to test transport reuse')
    url = str(self.get_url(self.bundle_name))
    self.read_mergeable_from_url(url)
    self.assertEqual(1, len(self.possible_transports))