from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
@staticmethod
def convert_module_args_to_api(parameters, exclusion=None):
    """
        Convert a list of string module args to API option format.
        For example, convert test_option to testOption.
        :param parameters: dict of parameters to be converted.
        :param exclusion: list of parameters to be ignored.
        :return: dict of key value pairs.
        """
    exclude_list = ['api_url', 'token_type', 'refresh_token', 'sa_secret_key', 'sa_client_id']
    if exclusion is not None:
        exclude_list += exclusion
    api_keys = {}
    for k, v in parameters.items():
        if k not in exclude_list:
            words = k.split('_')
            api_key = ''
            for word in words:
                if len(api_key) > 0:
                    word = word.title()
                api_key += word
            api_keys[api_key] = v
    return api_keys