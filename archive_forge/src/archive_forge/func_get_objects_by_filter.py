from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def get_objects_by_filter(self, operation_name, params):

    def match_filters(filter_params, obj):
        for k, v in iteritems(filter_params):
            if k not in obj or obj[k] != v:
                return False
        return True
    dummy, query_params, path_params = _get_user_params(params)
    url_params = {ParamName.QUERY_PARAMS: dict(query_params), ParamName.PATH_PARAMS: dict(path_params)}
    filters = params.get(ParamName.FILTERS) or {}
    if QueryParams.FILTER not in url_params[ParamName.QUERY_PARAMS] and 'name' in filters:
        url_params[ParamName.QUERY_PARAMS][QueryParams.FILTER] = self._stringify_name_filter(filters)
    item_generator = iterate_over_pageable_resource(partial(self.send_general_request, operation_name=operation_name), url_params)
    return (i for i in item_generator if match_filters(filters, i))