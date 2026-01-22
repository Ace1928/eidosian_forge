import ast
import copy
import re
import flask
import jsonschema
from oslo_config import cfg
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def assert_enabled_service_provider_object(service_provider):
    if service_provider.get('enabled') is not True:
        sp_id = service_provider['id']
        msg = 'Service Provider %(sp)s is disabled' % {'sp': sp_id}
        tr_msg = _('Service Provider %(sp)s is disabled') % {'sp': sp_id}
        LOG.debug(msg)
        raise exception.Forbidden(tr_msg)