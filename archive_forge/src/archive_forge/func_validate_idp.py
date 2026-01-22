import ast
import copy
import re
import flask
import jsonschema
from oslo_config import cfg
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def validate_idp(idp, protocol, assertion):
    """The IdP providing the assertion should be registered for the mapping."""
    remote_id_parameter = get_remote_id_parameter(idp, protocol)
    if not remote_id_parameter or not idp['remote_ids']:
        LOG.debug('Impossible to identify the IdP %s ', idp['id'])
        return
    try:
        idp_remote_identifier = assertion[remote_id_parameter]
    except KeyError:
        msg = _('Could not find Identity Provider identifier in environment')
        raise exception.ValidationError(msg)
    if idp_remote_identifier not in idp['remote_ids']:
        msg = _('Incoming identity provider identifier not included among the accepted identifiers.')
        raise exception.Forbidden(msg)