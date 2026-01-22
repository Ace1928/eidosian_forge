import datetime
import fixtures
import uuid
import freezegun
from oslo_config import fixture as config_fixture
from oslo_log import log
from keystone.common import fernet_utils
from keystone.common import utils as common_utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.server.flask import application
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import utils
def test_get_certificate_subject_dn_not_pem_format(self):
    self.assertRaises(exception.ValidationError, common_utils.get_certificate_subject_dn, 'MIIEkTCCAnkCFDIzsgpdRGF//5ukMuueXnRxQALhMA0GCSqGSIb3DQEBCwUAMIGC')