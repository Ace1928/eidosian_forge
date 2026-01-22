import functools
import re
import wsgiref.util
import http.client
from keystonemiddleware import auth_token
import oslo_i18n
from oslo_log import log
from oslo_serialization import jsonutils
import webob.dec
import webob.exc
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import render_token
from keystone.common import tokenless_auth
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.models import token_model
def fill_context(self, request):
    if authorization.AUTH_CONTEXT_ENV in request.environ:
        msg = 'Auth context already exists in the request environment; it will be used for authorization instead of creating a new one.'
        LOG.warning(msg)
        return
    kwargs = {'authenticated': False, 'overwrite': True}
    request_context = context.RequestContext.from_environ(request.environ, **kwargs)
    request.environ[context.REQUEST_CONTEXT_ENV] = request_context
    if request.environ.get(CONTEXT_ENV, {}).get('is_admin', False):
        request_context.is_admin = True
        auth_context = {}
    elif request.token_auth.has_user_token:
        if not self.token:
            self.token = PROVIDERS.token_provider_api.validate_token(request.user_token, access_rules_support=request.headers.get(authorization.ACCESS_RULES_HEADER))
        self._keystone_specific_values(self.token, request_context)
        request_context.auth_token = request.user_token
        auth_context = request_context.to_policy_values()
        additional = {'trust_id': request_context.trust_id, 'trustor_id': request_context.trustor_id, 'trustee_id': request_context.trustee_id, 'domain_id': request_context._domain_id, 'domain_name': request_context.domain_name, 'group_ids': request_context.group_ids, 'token': self.token}
        auth_context.update(additional)
    elif self._validate_trusted_issuer(request):
        auth_context = self._build_tokenless_auth_context(request)
        token_attributes = frozenset(('user_id', 'project_id', 'domain_id', 'user_domain_id', 'project_domain_id', 'user_domain_name', 'project_domain_name', 'roles', 'is_admin', 'project_name', 'domain_name', 'system_scope', 'is_admin_project', 'service_user_id', 'service_user_name', 'service_project_id', 'service_project_name', 'service_user_domain_idservice_user_domain_name', 'service_project_domain_id', 'service_project_domain_name', 'service_roles'))
        for attr in token_attributes:
            if attr in auth_context:
                setattr(request_context, attr, auth_context[attr])
        request_context.token_reference = {'token': None}
    else:
        return
    request_context.authenticated = True
    LOG.debug('RBAC: auth_context: %s', auth_context)
    request.environ[authorization.AUTH_CONTEXT_ENV] = auth_context