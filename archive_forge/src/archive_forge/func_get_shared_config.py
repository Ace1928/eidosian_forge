from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def get_shared_config(self, section_name, key):
    try:
        return self.shared_config[section_name][key]
    except KeyError:
        return None