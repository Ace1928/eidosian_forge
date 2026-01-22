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
def stack_domain_id(self):
    if not self._stack_domain_id:
        try:
            access = self.domain_admin_auth.get_access(self.session)
        except ks_exception.Unauthorized:
            LOG.error('Keystone client authentication failed')
            raise exception.AuthorizationFailure()
        self._stack_domain_id = access.domain_id
    return self._stack_domain_id