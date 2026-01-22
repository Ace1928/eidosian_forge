from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def get_fabric_id_details(name, all_fabrics):
    """
    obtain the fabric id using fabric name
    :param name: fabric name
    :param all_fabrics: All available fabric in the system
    :return: tuple
    1st item: fabric id
    2nd item: all details of fabric specified in dict
    """
    fabric_id, fabric_details = (None, None)
    for fabric_each in all_fabrics:
        if fabric_each['Name'] == name:
            fabric_id = fabric_each['Id']
            fabric_details = fabric_each
            break
    return (fabric_id, fabric_details)