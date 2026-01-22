from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.route_maps.route_maps import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.route_maps import (
def get_route_maps_data(self, connection):
    return connection.get('show running-config | section ^route-map')