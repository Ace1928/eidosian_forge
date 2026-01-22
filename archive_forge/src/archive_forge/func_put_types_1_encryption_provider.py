from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def put_types_1_encryption_provider(self, body, **kw):
    get_body = self.get_types_1_encryption()[2]
    for k, v in body.items():
        if k in get_body.keys():
            get_body.update([(k, v)])
    return (200, {}, get_body)