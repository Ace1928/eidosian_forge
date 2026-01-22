from octaviaclient.api import constants
from octaviaclient.api.v2 import octavia
from osc_lib import exceptions
from heat.engine.clients import client_plugin
from heat.engine import constraints
def get_l7policy(self, value):
    policy = self.client().find(path=constants.BASE_L7POLICY_URL, value=value, attr=DEFAULT_FIND_ATTR)
    return policy['id']