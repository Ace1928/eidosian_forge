import codecs
import io
import os
import os.path
import sys
import fixtures
from oslo_config import fixture as config
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import policy
def get_config_file_fullname(self, filename):
    return os.path.join(self.config_dir, filename.lstrip(os.sep))