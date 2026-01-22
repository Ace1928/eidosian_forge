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
def update_model_with_user_options(curr_model, update_model, module):
    """
    Update the 'update_model' with user provided values in 'module' for the specified 'attributes' if they are different
    from the values in the 'curr_model'.
    :param curr_model: A resource model instance representing the state of the current resource
    :param update_model: An instance of the update resource model for the current resource's type
    :param module: An AnsibleModule representing the options provided by the user
    :return: An updated 'update_model' instance filled with values that would need to be updated in the current resource
    state to satisfy the user's requested state.
    """
    attributes = update_model.attribute_map.keys()
    for attr in attributes:
        curr_value_for_attr = getattr(curr_model, attr, None)
        user_provided_value = _get_user_provided_value(module, attribute_name=attr)
        if curr_value_for_attr != user_provided_value:
            if user_provided_value is not None:
                _debug('User requested {0} for attribute {1}, whereas the current value is {2}. So adding it to the update model'.format(user_provided_value, attr, curr_value_for_attr))
                setattr(update_model, attr, user_provided_value)
            else:
                setattr(update_model, attr, curr_value_for_attr)
    return update_model