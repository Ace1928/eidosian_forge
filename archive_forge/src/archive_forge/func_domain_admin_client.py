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
def domain_admin_client(self):
    if not self._domain_admin_client:
        self._domain_admin_client = kc_v3.Client(session=self.session, auth=self.domain_admin_auth, connect_retries=cfg.CONF.client_retry_limit, interface=self._interface, region_name=self.auth_region_name)
    return self._domain_admin_client