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
def remove_item(self, info):
    if not isinstance(info, TemplateResourceInfo):
        return
    registry = self._registry
    for key in info.path[:-1]:
        registry = registry[key]
    if info.path[-1] in registry:
        registry.pop(info.path[-1])