from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
def set_element_attributes(self, source):
    """
            Return telemetry attributes for the current execution

            :param source: name of the module
            :type source: str
            :return: a dict containing telemetry attributes
        """
    attributes = {}
    attributes['config-mgmt'] = 'ansible'
    attributes['event-source'] = source
    return attributes