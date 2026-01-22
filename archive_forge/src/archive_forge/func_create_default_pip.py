from __future__ import absolute_import, division, print_function
import os
import re
import types
import copy
import inspect
import traceback
import json
from os.path import expanduser
from ansible.module_utils.basic import \
from ansible.module_utils.six.moves import configparser
import ansible.module_utils.six.moves.urllib.parse as urlparse
from base64 import b64encode, b64decode
from hashlib import sha256
from hmac import HMAC
from time import time
def create_default_pip(self, resource_group, location, public_ip_name, allocation_method='Dynamic', sku=None):
    """
        Create a default public IP address <public_ip_name> to associate with a network interface.
        If a PIP address matching <public_ip_name> exists, return it. Otherwise, create one.

        :param resource_group: name of an existing resource group
        :param location: a valid azure location
        :param public_ip_name: base name to assign the public IP address
        :param allocation_method: one of 'Static' or 'Dynamic'
        :param sku: sku
        :return: PIP object
        """
    pip = None
    self.log('Starting create_default_pip {0}'.format(public_ip_name))
    self.log('Check to see if public IP {0} exists'.format(public_ip_name))
    try:
        pip = self.network_client.public_ip_addresses.get(resource_group, public_ip_name)
    except Exception:
        pass
    if pip:
        self.log('Public ip {0} found.'.format(public_ip_name))
        self.check_provisioning_state(pip)
        return pip
    params = self.network_models.PublicIPAddress(location=location, public_ip_allocation_method=allocation_method, sku=sku)
    self.log('Creating default public IP {0}'.format(public_ip_name))
    try:
        poller = self.network_client.public_ip_addresses.begin_create_or_update(resource_group, public_ip_name, params)
    except Exception as exc:
        self.fail('Error creating {0} - {1}'.format(public_ip_name, str(exc)))
    return self.get_poller_result(poller)