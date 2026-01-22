from typing import List
from breezy import branch, urlutils
from breezy.tests import script
def test_first_use_no_option(self):
    self.do_command(*self.first_use_args)
    self.assertLocations(self.first_use_args)