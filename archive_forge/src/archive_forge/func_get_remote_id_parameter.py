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
def get_remote_id_parameter(idp, protocol):
    protocol_ref = PROVIDERS.federation_api.get_protocol(idp['id'], protocol)
    remote_id_parameter = protocol_ref.get('remote_id_attribute')
    if remote_id_parameter:
        return remote_id_parameter
    else:
        try:
            remote_id_parameter = CONF[protocol]['remote_id_attribute']
        except AttributeError:
            CONF.register_opt(cfg.StrOpt('remote_id_attribute'), group=protocol)
            try:
                remote_id_parameter = CONF[protocol]['remote_id_attribute']
            except AttributeError:
                pass
    if not remote_id_parameter:
        LOG.debug('Cannot find "remote_id_attribute" in configuration group %s. Trying default location in group federation.', protocol)
        remote_id_parameter = CONF.federation.remote_id_attribute
    return remote_id_parameter