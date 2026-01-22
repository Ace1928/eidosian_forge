import stevedore
from testtools import matchers
from keystonemiddleware.auth_token import _opts as new_opts
from keystonemiddleware import opts as old_opts
from keystonemiddleware.tests.unit import utils
def test_list_auth_token_opts(self):
    self._test_list_auth_token_opts(new_opts.list_opts())