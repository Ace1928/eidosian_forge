from unittest import mock
from troveclient.tests import fakes
from troveclient.tests.osc import utils
from troveclient.v1 import backups
from troveclient.v1 import clusters
from troveclient.v1 import configurations
from troveclient.v1 import databases
from troveclient.v1 import datastores
from troveclient.v1 import flavors
from troveclient.v1 import instances
from troveclient.v1 import limits
from troveclient.v1 import modules
from troveclient.v1 import quota
from troveclient.v1 import users
def get_non_absolute_limits(self):
    return limits.Limit(None, {'value': 200, 'verb': 'DELETE', 'remaining': 200, 'unit': 'MINUTE'})