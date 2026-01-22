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
def refresh_racks_lookup(self):
    url = self.api_endpoint + '/api/dcim/racks/?limit=0'
    racks = self.get_resource_list(api_url=url)
    self.racks_lookup = dict(((rack['id'], rack['name']) for rack in racks))

    def get_group_for_rack(rack):
        try:
            return (rack['id'], rack['group']['id'])
        except Exception:
            return (rack['id'], None)

    def get_role_for_rack(rack):
        try:
            return (rack['id'], rack['role']['slug'])
        except Exception:
            return (rack['id'], None)
    self.racks_group_lookup = dict(map(get_group_for_rack, racks))
    self.racks_role_lookup = dict(map(get_role_for_rack, racks))