from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
def normalize_interfaces_param(self):
    """ Goes through the interfaces parameter and gets it ready to be
        sent to the API. """
    for spec in self._module.params.get('interfaces') or ():
        if spec['addresses'] is None:
            del spec['addresses']
        if spec['network'] is None:
            del spec['network']
        for address in spec.get('addresses') or ():
            if address['address'] is None:
                del address['address']
            if address['subnet'] is None:
                del address['subnet']