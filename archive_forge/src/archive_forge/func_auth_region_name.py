import collections
import uuid
import weakref
from keystoneauth1 import exceptions as ks_exception
from keystoneauth1.identity import generic as ks_auth
from keystoneclient.v3 import client as kc_v3
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import importutils
from heat.common import config
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
@property
def auth_region_name(self):
    importutils.import_module('keystonemiddleware.auth_token')
    auth_region = cfg.CONF.keystone_authtoken.region_name
    if not auth_region:
        auth_region = self._region_name
    return auth_region