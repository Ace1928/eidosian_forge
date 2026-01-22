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
def check_and_create_resource(resource_type, create_fn, kwargs_create, list_fn, kwargs_list, module, model, existing_resources=None, exclude_attributes=None, dead_states=None, default_attribute_values=None, supports_sort_by_time_created=True):
    """
    This function checks whether there is a resource with same attributes as specified in the module options. If not,
    it creates and returns the resource.
    :param resource_type: Type of the resource to be created.
    :param create_fn: Function used in the module to handle create operation. The function should return a dict with
                      keys as resource & changed.
    :param kwargs_create: Dictionary of parameters for create operation.
    :param list_fn: List function in sdk to list all the resources of type resource_type.
    :param kwargs_list: Dictionary of parameters for list operation.
    :param module: Instance of AnsibleModule
    :param model: Model used to create a resource.
    :param exclude_attributes: The attributes which should not be used to distinguish the resource. e.g. display_name,
     dns_label.
    :param dead_states: List of states which can't transition to any of the usable states of the resource. This defaults
    to ["TERMINATING", "TERMINATED", "FAULTY", "FAILED", "DELETING", "DELETED", "UNKNOWN_ENUM_VALUE"]
    :param default_attribute_values: A dictionary containing default values for attributes.
    :return: A dictionary containing the resource & the "changed" status. e.g. {"vcn":{x:y}, "changed":True}
    """
    if module.params.get('force_create', None):
        _debug('Force creating {0}'.format(resource_type))
        result = call_with_backoff(create_fn, **kwargs_create)
        return result
    if exclude_attributes is None:
        exclude_attributes = {}
    if default_attribute_values is None:
        default_attribute_values = {}
    try:
        if existing_resources is None:
            if supports_sort_by_time_created:
                kwargs_list['sort_by'] = 'TIMECREATED'
            existing_resources = list_all_resources(list_fn, **kwargs_list)
    except ValueError:
        kwargs_list.pop('sort_by', None)
        try:
            existing_resources = list_all_resources(list_fn, **kwargs_list)
        except ServiceError as ex:
            module.fail_json(msg=ex.message)
    except ServiceError as ex:
        module.fail_json(msg=ex.message)
    result = dict()
    attributes_to_consider = _get_attributes_to_consider(exclude_attributes, model, module)
    if 'defined_tags' not in default_attribute_values:
        default_attribute_values['defined_tags'] = {}
    resource_matched = None
    _debug('Trying to find a match within {0} existing resources'.format(len(existing_resources)))
    for resource in existing_resources:
        if _is_resource_active(resource, dead_states):
            _debug("Comparing user specified values {0} against an existing resource's values {1}".format(module.params, to_dict(resource)))
            if does_existing_resource_match_user_inputs(to_dict(resource), module, attributes_to_consider, exclude_attributes, default_attribute_values):
                resource_matched = to_dict(resource)
                break
    if resource_matched:
        _debug('Resource with same attributes found: {0}.'.format(resource_matched))
        result[resource_type] = resource_matched
        result['changed'] = False
    else:
        _debug('No matching resource found. Attempting to create a new resource.')
        result = call_with_backoff(create_fn, **kwargs_create)
    return result