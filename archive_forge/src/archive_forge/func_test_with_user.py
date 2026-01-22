from .. import osutils, tests, urlutils
from ..directory_service import directories
from ..location import hooks as location_hooks
from ..location import location_to_url, rcp_location_to_url
def test_with_user(self):
    self.assertEqual('git+ssh://foo@example.com/srv/git/bar', rcp_location_to_url('foo@example.com:/srv/git/bar', scheme='git+ssh'))