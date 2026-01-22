import datetime
import uuid
from keystoneauth1 import fixture
from oslo_utils import timeutils
from keystoneclient import access
from keystoneclient.tests.unit import utils as test_utils
from keystoneclient.tests.unit.v3 import client_fixtures
from keystoneclient.tests.unit.v3 import utils
Check if is_federated property returns expected value.