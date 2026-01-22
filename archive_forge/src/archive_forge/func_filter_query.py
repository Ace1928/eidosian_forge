import abc
import codecs
import os.path
import random
import re
import sys
import uuid
import weakref
import ldap.controls
import ldap.filter
import ldappool
from oslo_log import log
from oslo_utils import reflection
from keystone.common import driver_hints
from keystone import exception
from keystone.i18n import _
def filter_query(self, hints, query=None):
    """Apply filtering to a query.

        :param hints: contains the list of filters, which may be None,
                      indicating that there are no filters to be applied.
                      If it's not None, then any filters satisfied here will be
                      removed so that the caller will know if any filters
                      remain to be applied.
        :param query: LDAP query into which to include filters

        :returns query: LDAP query, updated with any filters satisfied

        """

    def build_filter(filter_):
        """Build a filter for the query.

            :param filter_: the dict that describes this filter

            :returns query: LDAP query term to be added

            """
        ldap_attr = self.attribute_mapping[filter_['name']]
        val_esc = ldap.filter.escape_filter_chars(filter_['value'])
        if filter_['case_sensitive']:
            return
        if filter_['name'] == 'enabled':
            return
        if filter_['comparator'] == 'equals':
            query_term = u'(%(attr)s=%(val)s)' % {'attr': ldap_attr, 'val': val_esc}
        elif filter_['comparator'] == 'contains':
            query_term = u'(%(attr)s=*%(val)s*)' % {'attr': ldap_attr, 'val': val_esc}
        elif filter_['comparator'] == 'startswith':
            query_term = u'(%(attr)s=%(val)s*)' % {'attr': ldap_attr, 'val': val_esc}
        elif filter_['comparator'] == 'endswith':
            query_term = u'(%(attr)s=*%(val)s)' % {'attr': ldap_attr, 'val': val_esc}
        else:
            return
        return query_term
    if query is None:
        query = ''
    if hints is None:
        return query
    filter_list = []
    satisfied_filters = []
    for filter_ in hints.filters:
        if filter_['name'] not in self.attribute_mapping:
            continue
        new_filter = build_filter(filter_)
        if new_filter is not None:
            filter_list.append(new_filter)
            satisfied_filters.append(filter_)
    if filter_list:
        query = u'(&%s%s)' % (query, ''.join(filter_list))
    for filter_ in satisfied_filters:
        hints.filters.remove(filter_)
    return query