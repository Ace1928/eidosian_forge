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
def get_assertion_params_from_env():
    LOG.debug('Environment variables: %s', flask.request.environ)
    prefix = CONF.federation.assertion_prefix
    for k, v in list(flask.request.environ.items()):
        if not k.startswith(prefix):
            continue
        if not isinstance(v, str) and getattr(v, 'decode', False):
            v = v.decode('ISO-8859-1')
        yield (k, v)