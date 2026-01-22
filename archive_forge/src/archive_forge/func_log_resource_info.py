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
def log_resource_info(self, show_all=False, prefix=None):
    registry = self._registry
    prefix = '%s ' % prefix if prefix is not None else ''
    for name in registry:
        if name == 'resources':
            continue
        if show_all or isinstance(registry[name], TemplateResourceInfo):
            msg = '%(p)sRegistered: %(t)s' % {'p': prefix, 't': str(registry[name])}
            LOG.info(msg)