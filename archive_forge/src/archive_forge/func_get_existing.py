from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def get_existing(self):
    """
        This method is used to get the existing object(s) based on the path specified in the module. Each module should
        build the URL so that if the object's name is supplied, then it will retrieve the configuration for that particular
        object, but if no name is supplied, then it will retrieve all MOs for the class. Following this method will ensure
        that this method can be used to supply the existing configuration when using the get_diff method. The response, status,
        and existing configuration will be added to the self.result dictionary.
        """
    uri = self.url + self.filter_string
    self.api_call('GET', uri, data=None, return_response=False)