from __future__ import absolute_import, division, print_function
import json
import uuid
import math
import os
import datetime
from copy import deepcopy
from functools import partial
from sys import version as python_version
from threading import Thread
from typing import Iterable
from itertools import chain
from collections import defaultdict
from ipaddress import ip_interface
from ansible.constants import DEFAULT_LOCAL_TMP
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib import error as urllib_error
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six import raise_from
def refresh_ipaddresses(self):
    url = self.api_endpoint + '/api/ipam/ip-addresses/?limit=0&assigned_to_interface=true'
    ipaddresses = []
    if self.fetch_all:
        ipaddresses = self.get_resource_list(url)
    else:
        device_ips = self.get_resource_list_chunked(api_url=url, query_key='device_id', query_values=list(self.devices_with_ips))
        vm_ips = self.get_resource_list_chunked(api_url=url, query_key='virtual_machine_id', query_values=self.vms_lookup.keys())
        ipaddresses = chain(device_ips, vm_ips)
    self.ipaddresses_intf_lookup = defaultdict(dict)
    self.ipaddresses_lookup = defaultdict(dict)
    self.vm_ipaddresses_intf_lookup = defaultdict(dict)
    self.vm_ipaddresses_lookup = defaultdict(dict)
    self.device_ipaddresses_intf_lookup = defaultdict(dict)
    self.device_ipaddresses_lookup = defaultdict(dict)
    for ipaddress in ipaddresses:
        if ipaddress.get('assigned_object_id'):
            interface_id = ipaddress['assigned_object_id']
            ip_id = ipaddress['id']
            ipaddress_copy = ipaddress.copy()
            if ipaddress['assigned_object_type'] == 'virtualization.vminterface':
                self.vm_ipaddresses_lookup[ip_id] = ipaddress_copy
                self.vm_ipaddresses_intf_lookup[interface_id][ip_id] = ipaddress_copy
            else:
                self.device_ipaddresses_lookup[ip_id] = ipaddress_copy
                self.device_ipaddresses_intf_lookup[interface_id][ip_id] = ipaddress_copy
            del ipaddress_copy['assigned_object_id']
            del ipaddress_copy['assigned_object_type']
            del ipaddress_copy['assigned_object']
            continue
        if not ipaddress.get('interface'):
            continue
        interface_id = ipaddress['interface']['id']
        ip_id = ipaddress['id']
        ipaddress_copy = ipaddress.copy()
        self.ipaddresses_intf_lookup[interface_id][ip_id] = ipaddress_copy
        self.ipaddresses_lookup[ip_id] = ipaddress_copy
        del ipaddress_copy['interface']