from __future__ import absolute_import, division, print_function
import os
import re
import logging
import sys
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
def ref_n_str_cmp(x, y):
    """
    compares two references
    1. check for exact reference
    2. check for obj_type/uuid
    3. check for name

    if x is ref=name then extract uuid and name from y and use it.
    if x is http_ref then
        strip x and y
        compare them.

    if x and y are urls then match with split on #
    if x is a RE_REF_MATCH then extract name
    if y is a REF_MATCH then extract name
    :param x: first string
    :param y: second string from controller's object

    Returns
        True if they are equivalent else False
    """
    if type(y) in (int, float, bool, int, complex):
        y = str(y)
        x = str(x)
    if not (_check_type_string(x) and _check_type_string(y)):
        return False
    y_uuid = y_name = str(y)
    x = str(x)
    if RE_REF_MATCH.match(x):
        x = x.split('name=')[1]
    elif HTTP_REF_MATCH.match(x):
        x = x.rsplit('#', 1)[0]
        y = y.rsplit('#', 1)[0]
    elif RE_REF_MATCH.match(y):
        y = y.split('name=')[1]
    if HTTP_REF_W_NAME_MATCH.match(y):
        path = y.split('api/', 1)[1]
        uuid_or_name = path.split('/')[-1]
        parts = uuid_or_name.rsplit('#', 1)
        y_uuid = parts[0]
        y_name = parts[1] if len(parts) > 1 else ''
    result = x in (y, y_name, y_uuid)
    if not result:
        log.debug('x: %s y: %s y_name %s y_uuid %s', x, y, y_name, y_uuid)
    return result