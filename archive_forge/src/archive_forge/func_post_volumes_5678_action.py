from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_volumes_5678_action(self, body, **kw):
    return self.post_volumes_1234_action(body, **kw)