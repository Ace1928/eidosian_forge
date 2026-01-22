import collections
import fnmatch
import glob
import itertools
import os.path
import re
import weakref
from oslo_config import cfg
from oslo_log import log
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.engine import support
def get_class_to_instantiate(self, resource_type, resource_name=None):
    return self.registry.get_class_to_instantiate(resource_type, resource_name)