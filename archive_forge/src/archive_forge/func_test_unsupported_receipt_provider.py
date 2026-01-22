import datetime
import uuid
from oslo_utils import timeutils
import freezegun
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.models import receipt_model
from keystone import receipt
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
def test_unsupported_receipt_provider(self):
    self.config_fixture.config(group='receipt', provider='MyProvider')
    self.assertRaises(ImportError, receipt.provider.Manager)