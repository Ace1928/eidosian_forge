from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def delete_volumes_1234_metadata_key2(self, **kw):
    return (204, {}, None)