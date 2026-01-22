from the request environment and it's identified by the ``swift.cache`` key.
import copy
import re
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from keystoneauth1.loading import session as session_loading
import oslo_cache
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import webob.dec
from keystonemiddleware._common import config
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.auth_token import _base
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.auth_token import _identity
from keystonemiddleware.auth_token import _opts
from keystonemiddleware.auth_token import _request
from keystonemiddleware.auth_token import _user_plugin
from keystonemiddleware.i18n import _
def validate_allowed_request(self, request, token):
    self.log.debug('Validating token access rules against request')
    app_cred = token.get('application_credential')
    if not app_cred:
        return
    access_rules = app_cred.get('access_rules')
    if access_rules is None:
        return
    if hasattr(self, '_conf'):
        my_service_type = self._conf.get('service_type')
    else:
        my_service_type = self._service_type
    if not my_service_type:
        self.log.warning('Cannot validate request with restricted access rules. Set service_type in [keystone_authtoken] to allow access rule validation.')
        raise ksm_exceptions.InvalidToken(_('Token authorization failed'))
    if my_service_type == 'identity' and request.method == 'GET' and request.path.endswith('/v3/auth/tokens'):
        return
    catalog = token['catalog']
    catalog_svcs = [s for s in catalog if s['type'] == my_service_type]
    if len(catalog_svcs) == 0:
        self.log.warning('Cannot validate request with restricted access rules. service_type in [keystone_authtoken] is not a valid service type in the catalog.')
        raise ksm_exceptions.InvalidToken(_('Token authorization failed'))
    if request.service_token:
        return
    for access_rule in access_rules:
        method = access_rule['method']
        path = access_rule['path']
        service = access_rule['service']
        if request.method == method and service == my_service_type and _path_matches(request.path, path):
            return
    raise ksm_exceptions.InvalidToken(_('Token authorization failed'))