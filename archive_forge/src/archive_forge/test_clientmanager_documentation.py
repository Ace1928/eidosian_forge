import copy
from keystoneauth1 import token_endpoint
from osc_lib.tests import utils as osc_lib_test_utils
from openstackclient.common import clientmanager
from openstackclient.tests.unit import fakes
Allow subclasses to override the ClientManager class