import logging
import warnings
from oslo_serialization import jsonutils
from keystoneclient.auth.identity import v3 as v3_auth
from keystoneclient import exceptions
from keystoneclient import httpclient
from keystoneclient.i18n import _
from keystoneclient.v3 import access_rules
from keystoneclient.v3 import application_credentials
from keystoneclient.v3 import auth
from keystoneclient.v3.contrib import endpoint_filter
from keystoneclient.v3.contrib import endpoint_policy
from keystoneclient.v3.contrib import federation
from keystoneclient.v3.contrib import oauth1
from keystoneclient.v3.contrib import simple_cert
from keystoneclient.v3.contrib import trusts
from keystoneclient.v3 import credentials
from keystoneclient.v3 import domain_configs
from keystoneclient.v3 import domains
from keystoneclient.v3 import ec2
from keystoneclient.v3 import endpoint_groups
from keystoneclient.v3 import endpoints
from keystoneclient.v3 import groups
from keystoneclient.v3 import limits
from keystoneclient.v3 import policies
from keystoneclient.v3 import projects
from keystoneclient.v3 import regions
from keystoneclient.v3 import registered_limits
from keystoneclient.v3 import role_assignments
from keystoneclient.v3 import roles
from keystoneclient.v3 import services
from keystoneclient.v3 import tokens
from keystoneclient.v3 import users
def process_token(self, **kwargs):
    """Extract and process information from the new auth_ref.

        And set the relevant authentication information.
        """
    super(Client, self).process_token(**kwargs)
    if self.auth_ref.domain_scoped:
        if not self.auth_ref.domain_id:
            raise exceptions.AuthorizationFailure(_("Token didn't provide domain_id"))
        self._process_management_url(kwargs.get('region_name'))
        self.domain_name = self.auth_ref.domain_name
        self.domain_id = self.auth_ref.domain_id
    if self._management_url:
        self._management_url = self._management_url.replace('/v2.0', '/v3')