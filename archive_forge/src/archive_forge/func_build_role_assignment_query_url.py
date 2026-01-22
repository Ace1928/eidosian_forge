import datetime
import http.client
import oslo_context.context
from oslo_serialization import jsonutils
from testtools import matchers
import uuid
import webtest
from keystone.common import authorization
from keystone.common import cache
from keystone.common import provider_api
from keystone.common.validation import validators
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import rest
def build_role_assignment_query_url(self, effective=False, **filters):
    """Build and return a role assignment query url with provided params.

        Available filters are: domain_id, project_id, user_id, group_id,
        role_id and inherited_to_projects.
        """
    query_params = '?effective' if effective else ''
    for k, v in filters.items():
        query_params += '?' if not query_params else '&'
        if k == 'inherited_to_projects':
            query_params += 'scope.OS-INHERIT:inherited_to=projects'
        else:
            if k in ['domain_id', 'project_id']:
                query_params += 'scope.'
            elif k not in ['user_id', 'group_id', 'role_id']:
                raise ValueError("Invalid key '%s' in provided filters." % k)
            query_params += '%s=%s' % (k.replace('_', '.'), v)
    return '/role_assignments%s' % query_params