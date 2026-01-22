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
def refresh_rack_groups_lookup(self):
    if self.api_version >= version.parse('2.11'):
        return
    url = self.api_endpoint + '/api/dcim/rack-groups/?limit=0'
    rack_groups = self.get_resource_list(api_url=url)
    self.rack_groups_lookup = dict(((rack_group['id'], rack_group['slug']) for rack_group in rack_groups))

    def get_rack_group_parent(rack_group):
        try:
            return (rack_group['id'], rack_group['parent']['id'])
        except Exception:
            return (rack_group['id'], None)
    self.rack_group_parent_lookup = dict(map(get_rack_group_parent, rack_groups))