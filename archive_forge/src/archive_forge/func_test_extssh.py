from .. import osutils, tests, urlutils
from ..directory_service import directories
from ..location import hooks as location_hooks
from ..location import location_to_url, rcp_location_to_url
def test_extssh(self):
    self.assertEqual('cvs+ssh://anonymous@odessa.cvs.sourceforge.net/cvsroot/odess', location_to_url(':extssh:anonymous@odessa.cvs.sourceforge.net:/cvsroot/odess'))