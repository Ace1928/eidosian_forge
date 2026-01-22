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
def create_attribute_mapping_rules_processor(mapping):
    version = mapping.get('schema_version', get_default_attribute_mapping_schema_version())
    return IDP_ATTRIBUTE_MAPPING_SCHEMAS[version]['processor'](mapping['id'], mapping['rules'])