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
def register_stack_lifecycle_plugin(self, stack_lifecycle_name, stack_lifecycle_class):
    self.stack_lifecycle_plugins.append((stack_lifecycle_name, stack_lifecycle_class))