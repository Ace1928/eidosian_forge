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
def does_existing_resource_match_user_inputs(existing_resource, module, attributes_to_compare, exclude_attributes, default_attribute_values=None):
    """
    Check if 'attributes_to_compare' in an existing_resource match the desired state provided by a user in 'module'.
    :param existing_resource: A dictionary representing an existing resource's values.
    :param module: The AnsibleModule representing the options provided by the user.
    :param attributes_to_compare: A list of attributes of a resource that are used to compare if an existing resource
                                    matches the desire state of the resource expressed by the user in 'module'.
    :param exclude_attributes: The attributes, that a module author provides, which should not be used to match the
        resource. This dictionary typically includes: (a) attributes which are initialized with dynamic default values
        like 'display_name', 'security_list_ids' for subnets and (b) attributes that don't have any defaults like
        'dns_label' in VCNs. The attributes are part of keys and 'True' is the value for all existing keys.
    :param default_attribute_values: A dictionary containing default values for attributes.
    :return: True if the values for the list of attributes is the same in the existing_resource and module instances.
    """
    if not default_attribute_values:
        default_attribute_values = {}
    for attr in attributes_to_compare:
        attribute_with_default_metadata = None
        if attr in existing_resource:
            resources_value_for_attr = existing_resource[attr]
            user_provided_value_for_attr = _get_user_provided_value(module, attr)
            if user_provided_value_for_attr is not None:
                res = [True]
                check_if_user_value_matches_resources_attr(attr, resources_value_for_attr, user_provided_value_for_attr, exclude_attributes, default_attribute_values, res)
                if not res[0]:
                    _debug("Mismatch on attribute '{0}'. User provided value is {1} & existing resource's valueis {2}.".format(attr, user_provided_value_for_attr, resources_value_for_attr))
                    return False
            elif exclude_attributes.get(attr) is None and resources_value_for_attr is not None:
                if module.argument_spec.get(attr):
                    attribute_with_default_metadata = module.argument_spec.get(attr)
                    default_attribute_value = attribute_with_default_metadata.get('default', None)
                    if default_attribute_value is not None:
                        if existing_resource[attr] != default_attribute_value:
                            return False
                    elif not is_attr_assigned_default(default_attribute_values, attr, existing_resource[attr]):
                        return False
        else:
            _debug("Attribute {0} is in the create model of resource {1}but doesn't exist in the get model of the resource".format(attr, existing_resource.__class__))
    return True