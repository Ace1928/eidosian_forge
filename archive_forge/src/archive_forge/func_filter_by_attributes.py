import abc
import collections
import functools
import re
import uuid
import wsgiref.util
import flask
from flask import blueprints
import flask_restful
import flask_restful.utils
import http.client
from oslo_log import log
from oslo_log import versionutils
from oslo_serialization import jsonutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import driver_hints
from keystone.common import json_home
from keystone.common.rbac_enforcer import enforcer
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
@classmethod
def filter_by_attributes(cls, refs, hints):
    """Filter a list of references by filter values."""

    def _attr_match(ref_attr, val_attr):
        """Matche attributes allowing for booleans as strings.

            We test explicitly for a value that defines it as 'False',
            which also means that the existence of the attribute with
            no value implies 'True'

            """
        if type(ref_attr) is bool:
            return ref_attr == utils.attr_as_boolean(val_attr)
        else:
            return ref_attr == val_attr

    def _inexact_attr_match(inexact_filter, ref):
        """Apply an inexact filter to a result dict.

            :param inexact_filter: the filter in question
            :param ref: the dict to check

            :returns: True if there is a match

            """
        comparator = inexact_filter['comparator']
        key = inexact_filter['name']
        if key in ref:
            filter_value = inexact_filter['value']
            target_value = ref[key]
            if not inexact_filter['case_sensitive']:
                filter_value = filter_value.lower()
                target_value = target_value.lower()
            if comparator == 'contains':
                return filter_value in target_value
            elif comparator == 'startswith':
                return target_value.startswith(filter_value)
            elif comparator == 'endswith':
                return target_value.endswith(filter_value)
            else:
                return True
        return False
    for f in hints.filters:
        if f['comparator'] == 'equals':
            attr = f['name']
            value = f['value']
            refs = [r for r in refs if _attr_match(utils.flatten_dict(r).get(attr), value)]
        else:
            refs = [r for r in refs if _inexact_attr_match(f, r)]
    return refs