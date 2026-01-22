import uuid
import oauthlib.common
from oauthlib import oauth1
from oslo_log import log
from keystone.common import manager
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def get_oauth_headers(headers):
    parameters = {}
    if headers and 'Authorization' in headers:
        auth_header = headers['Authorization']
        params = oauth1.rfc5849.utils.parse_authorization_header(auth_header)
        parameters.update(dict(params))
        return parameters
    else:
        msg = 'Cannot retrieve Authorization headers'
        LOG.error(msg)
        raise exception.OAuthHeadersMissingError()