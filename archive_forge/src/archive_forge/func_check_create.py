from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
def check_create(self):
    self._set_default_ip_protocol()
    self._set_default_profiles()
    self._override_port_by_type()
    self._override_protocol_by_type()
    self._verify_type_has_correct_profiles()
    self._verify_default_persistence_profile_for_type()
    self._verify_fallback_persistence_profile_for_type()
    self._update_persistence_profile()
    self._verify_virtual_has_required_parameters()
    self._ensure_server_type_supports_vlans()
    self._override_vlans_if_all_specified()
    self._check_source_and_destination_match()
    self._verify_type_has_correct_ip_protocol()
    self._verify_minimum_profile()
    self._verify_dhcp_profile()
    self._verify_fastl4_profile()
    self._verify_stateless_profile_on_create()