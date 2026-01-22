from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def put_v3_default_types_629632e7_99d2_4c40_9ae3_106fa3b1c9b7(self, **kwargs):
    default_type = stub_default_type()
    return (200, {}, default_type)