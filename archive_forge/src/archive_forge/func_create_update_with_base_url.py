from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def create_update_with_base_url(self, url, **kwargs):
    return self._cs_request(url, 'PUT', **kwargs)