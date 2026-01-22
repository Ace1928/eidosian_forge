from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
def modify_metadef_namespace(self):
    self._enforce('modify_metadef_namespace')