from __future__ import absolute_import, division, print_function
import re
from ansible.errors import AnsibleFilterError
replaces specific keys with mentioned after data"
    :param data: The data passed in (data|replace_keys(...))
    :type data: raw
    :param target: List of keys on with operation is to be performed
    :type data: list
    :type elements: string
    :param matching_parameter: matching type of the target keys with data keys
    :type data: list
    :type elements: dict
    