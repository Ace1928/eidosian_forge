import uuid
import oauthlib.common
from oauthlib import oauth1
from oslo_log import log
from keystone.common import manager
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def validate_oauth_params(query_string):
    params = oauthlib.common.extract_params(query_string)
    params_fitered = {k: v for k, v in params if not k.startswith('oauth_')}
    if params_fitered:
        if 'error' in params_fitered:
            msg = 'Validation failed with errors: %(error)s, detail message is: %(desc)s.' % {'error': params_fitered['error'], 'desc': params_fitered['error_description']}
            tr_msg = _('Validation failed with errors: %(error)s, detail message is: %(desc)s.') % {'error': params_fitered['error'], 'desc': params_fitered['error_description']}
        else:
            msg = 'Unknown parameters found,please provide only oauth parameters.'
            tr_msg = _('Unknown parameters found,please provide only oauth parameters.')
        LOG.warning(msg)
        raise exception.ValidationError(message=tr_msg)