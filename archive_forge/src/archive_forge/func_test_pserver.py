from .. import osutils, tests, urlutils
from ..directory_service import directories
from ..location import hooks as location_hooks
from ..location import location_to_url, rcp_location_to_url
def test_pserver(self):
    self.assertEqual('cvs+pserver://anonymous@odessa.cvs.sourceforge.net/cvsroot/odess', location_to_url(':pserver:anonymous@odessa.cvs.sourceforge.net:/cvsroot/odess'))
    self.assertRaises(urlutils.InvalidURL, location_to_url, ':pserver:blah')
    self.assertRaises(urlutils.InvalidURL, location_to_url, ':pserver:blah:bloe')