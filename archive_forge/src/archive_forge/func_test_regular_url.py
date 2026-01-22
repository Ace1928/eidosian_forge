from .. import osutils, tests, urlutils
from ..directory_service import directories
from ..location import hooks as location_hooks
from ..location import location_to_url, rcp_location_to_url
def test_regular_url(self):
    self.assertEqual('file://foo', location_to_url('file://foo'))