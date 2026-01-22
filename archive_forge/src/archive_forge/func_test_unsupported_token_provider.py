import datetime
from oslo_utils import timeutils
import urllib
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.models import token_model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone import token
from keystone.token import provider
def test_unsupported_token_provider(self):
    self.config_fixture.config(group='token', provider='MyProvider')
    self.assertRaises(ImportError, token.provider.Manager)