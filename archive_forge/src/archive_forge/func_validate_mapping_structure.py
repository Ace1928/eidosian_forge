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
def validate_mapping_structure(ref):
    version = ref.get('schema_version', get_default_attribute_mapping_schema_version())
    LOG.debug('Validating mapping [%s] using validator from version [%s].', ref, version)
    v = jsonschema.Draft4Validator(IDP_ATTRIBUTE_MAPPING_SCHEMAS[version]['schema'])
    messages = ''
    for error in sorted(v.iter_errors(ref), key=str):
        messages = messages + error.message + '\n'
    if messages:
        raise exception.ValidationError(messages)