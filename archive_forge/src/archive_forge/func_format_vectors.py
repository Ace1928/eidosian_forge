from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def format_vectors(self, vectors):
    result = None
    for x in vectors:
        vector = ApiParameters(params=x)
        self.vectors[vector.name] = x
        if vector.name == self.want.name:
            result = vector
    if not result:
        return ApiParameters()
    return result