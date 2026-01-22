import logging
import warnings
from keystoneclient.auth.identity import v2 as v2_auth
from keystoneclient import exceptions
from keystoneclient import httpclient
from keystoneclient.i18n import _
from keystoneclient.v2_0 import certificates
from keystoneclient.v2_0 import ec2
from keystoneclient.v2_0 import endpoints
from keystoneclient.v2_0 import extensions
from keystoneclient.v2_0 import roles
from keystoneclient.v2_0 import services
from keystoneclient.v2_0 import tenants
from keystoneclient.v2_0 import tokens
from keystoneclient.v2_0 import users
def get_raw_token_from_identity_service(self, auth_url, username=None, password=None, tenant_name=None, tenant_id=None, token=None, project_name=None, project_id=None, trust_id=None, **kwargs):
    """Authenticate against the v2 Identity API.

        If a token is provided it will be used in preference over username and
        password.

        :returns: access.AccessInfo if authentication was successful.
        :raises keystoneclient.exceptions.AuthorizationFailure: if unable to
            authenticate or validate the existing authorization token
        """
    try:
        if auth_url is None:
            raise ValueError(_('Cannot authenticate without an auth_url'))
        new_kwargs = {'trust_id': trust_id, 'tenant_id': project_id or tenant_id, 'tenant_name': project_name or tenant_name}
        if token:
            plugin = v2_auth.Token(auth_url, token, **new_kwargs)
        elif username and password:
            plugin = v2_auth.Password(auth_url, username, password, **new_kwargs)
        else:
            msg = _('A username and password or token is required.')
            raise exceptions.AuthorizationFailure(msg)
        return plugin.get_auth_ref(self.session)
    except (exceptions.AuthorizationFailure, exceptions.Unauthorized):
        _logger.debug('Authorization Failed.')
        raise
    except exceptions.EndpointNotFound:
        msg = _('There was no suitable authentication url for this request')
        raise exceptions.AuthorizationFailure(msg)
    except Exception as e:
        raise exceptions.AuthorizationFailure(_('Authorization Failed: %s') % e)