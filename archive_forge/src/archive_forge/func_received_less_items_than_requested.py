from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def received_less_items_than_requested(items_in_response, items_expected):
    if items_in_response == items_expected:
        return False
    elif items_in_response < items_expected:
        return True
    raise FtdUnexpectedResponse('Get List of Objects Response from the server contains more objects than requested. There are {0} item(s) in the response while {1} was(ere) requested'.format(items_in_response, items_expected))