from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def create_or_update_resource_and_wait(resource_type, function, kwargs_function, module, wait_applicable, get_fn, get_param, states, client, update_target_resource_id_in_get_param=False, kwargs_get=None):
    """
    A utility function to create or update a resource and wait for the resource to get into the state as specified in
    the module options.
    :param resource_type: Type of the resource to be created. e.g. "vcn"
    :param function: Function in the SDK to create or update the resource.
    :param kwargs_function: Dictionary containing arguments to be used to call the create or update function
    :param module: Instance of AnsibleModule.
    :param wait_applicable: Specifies if wait for create is applicable for this resource
    :param get_fn: Function in the SDK to get the resource. e.g. virtual_network_client.get_vcn
    :param get_param: Name of the argument in the SDK get function. e.g. "vcn_id"
    :param states: List of lifecycle states to watch for while waiting after create_fn is called.
                   e.g. [module.params['wait_until'], "FAULTY"]
    :param client: OCI service client instance to call the service periodically to retrieve data.
                   e.g. VirtualNetworkClient()
    :param kwargs_get: Dictionary containing arguments to be used to call the get function which requires multiple arguments.
    :return: A dictionary containing the resource & the "changed" status. e.g. {"vcn":{x:y}, "changed":True}
    """
    result = create_resource(resource_type, function, kwargs_function, module)
    resource = result[resource_type]
    result[resource_type] = wait_for_resource_lifecycle_state(client, module, wait_applicable, kwargs_get, get_fn, get_param, resource, states, resource_type)
    return result