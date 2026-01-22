from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def is_cvo_update_needed(self, rest_api, headers, parameters, changeable_params, provider):
    modify, error = self.get_modify_cvo_params(rest_api, headers, parameters, provider)
    if error is not None:
        return (None, error)
    unmodifiable = [attr for attr in modify if attr not in changeable_params]
    if unmodifiable:
        return (None, '%s cannot be modified.' % str(unmodifiable))
    return (modify, None)