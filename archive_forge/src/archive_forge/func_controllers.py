from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat, pprint
import time
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
@property
def controllers(self):
    """Retrieve a mapping of controller labels to their references
        {
            'A': '070000000000000000000001',
            'B': '070000000000000000000002',
        }
        :return: the controllers defined on the system
        """
    try:
        rc, controllers = request(self.url + 'storage-systems/%s/controllers' % self.ssid, headers=HEADERS, **self.creds)
    except Exception as err:
        controllers = list()
        self.module.fail_json(msg='Failed to retrieve the controller settings. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
    controllers.sort(key=lambda c: c['physicalLocation']['slot'])
    controllers_dict = dict()
    i = ord('A')
    for controller in controllers:
        label = chr(i)
        settings = dict(controllerSlot=controller['physicalLocation']['slot'], controllerRef=controller['controllerRef'], ssh=controller['networkSettings']['remoteAccessEnabled'])
        controllers_dict[label] = settings
        i += 1
    return controllers_dict