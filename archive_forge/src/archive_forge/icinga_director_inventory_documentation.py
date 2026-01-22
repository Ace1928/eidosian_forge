from __future__ import absolute_import, division, print_function
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.urls import open_url
import json
Return dynamic inventory from source