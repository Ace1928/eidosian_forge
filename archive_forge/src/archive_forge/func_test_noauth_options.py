from keystoneauth1.loading._plugins import noauth as loader
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_noauth_options(self):
    opts = loader.NoAuth().get_options()
    self.assertEqual(['endpoint'], [o.name for o in opts])