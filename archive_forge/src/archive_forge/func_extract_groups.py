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
def extract_groups(self, groups_by_domain):
    for groups in list(groups_by_domain.values()):
        for group in list({g['name']: g for g in groups}.values()):
            yield group