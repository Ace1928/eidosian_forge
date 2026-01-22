from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
def get_setval_path(module_or_path_data):
    """Build setval for path parameter based on playbook inputs
    Full Command:
      - path {name} depth {depth} query-condition {query_condition} filter-condition {filter_condition}
    Required:
      - path {name}
    Optional:
      - depth {depth}
      - query-condition {query_condition},
      - filter-condition {filter_condition}
    """
    if isinstance(module_or_path_data, dict):
        path = module_or_path_data
    else:
        path = module_or_path_data.params['config']['sensor_groups'][0].get('path')
    if path is None:
        return path
    setval = 'path {name}'
    if 'depth' in path.keys():
        if path.get('depth') != 'None':
            setval = setval + ' depth {depth}'
    if 'query_condition' in path.keys():
        if path.get('query_condition') != 'None':
            setval = setval + ' query-condition {query_condition}'
    if 'filter_condition' in path.keys():
        if path.get('filter_condition') != 'None':
            setval = setval + ' filter-condition {filter_condition}'
    return setval