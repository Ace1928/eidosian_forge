from __future__ import (absolute_import, division, print_function)
import atexit
import datetime
import itertools
import json
import os
import re
import ssl
import sys
import uuid
from time import time
from jinja2 import Environment
from ansible.module_utils.six import integer_types, PY3
from ansible.module_utils.six.moves import configparser
def instances_to_inventory(self, instances):
    """ Convert a list of vm objects into a json compliant inventory """
    self.debugl('re-indexing instances based on ini settings')
    inventory = VMWareInventory._empty_inventory()
    inventory['all'] = {}
    inventory['all']['hosts'] = []
    for idx, instance in enumerate(instances):
        thisid = str(uuid.uuid4())
        idata = instance[1]
        inventory['all']['hosts'].append(thisid)
        inventory['_meta']['hostvars'][thisid] = idata.copy()
        inventory['_meta']['hostvars'][thisid]['ansible_uuid'] = thisid
    name_mapping = self.create_template_mapping(inventory, self.config.get('vmware', 'alias_pattern'))
    host_mapping = self.create_template_mapping(inventory, self.config.get('vmware', 'host_pattern'))
    for k, v in name_mapping.items():
        if not host_mapping or k not in host_mapping:
            continue
        try:
            inventory['_meta']['hostvars'][k]['ansible_host'] = host_mapping[k]
            inventory['_meta']['hostvars'][k]['ansible_ssh_host'] = host_mapping[k]
        except Exception:
            continue
        if k == v:
            continue
        inventory['all']['hosts'].append(v)
        inventory['_meta']['hostvars'][v] = inventory['_meta']['hostvars'][k]
        inventory['all']['hosts'].remove(k)
        inventory['_meta']['hostvars'].pop(k, None)
    self.debugl('pre-filtered hosts:')
    for i in inventory['all']['hosts']:
        self.debugl('  * %s' % i)
    for hf in self.host_filters:
        if not hf:
            continue
        self.debugl('filter: %s' % hf)
        filter_map = self.create_template_mapping(inventory, hf, dtype='boolean')
        for k, v in filter_map.items():
            if not v:
                inventory['all']['hosts'].remove(k)
                inventory['_meta']['hostvars'].pop(k, None)
    self.debugl('post-filter hosts:')
    for i in inventory['all']['hosts']:
        self.debugl('  * %s' % i)
    for gbp in self.groupby_patterns:
        groupby_map = self.create_template_mapping(inventory, gbp)
        for k, v in groupby_map.items():
            if v not in inventory:
                inventory[v] = {}
                inventory[v]['hosts'] = []
            if k not in inventory[v]['hosts']:
                inventory[v]['hosts'].append(k)
    if self.config.get('vmware', 'groupby_custom_field'):
        for k, v in inventory['_meta']['hostvars'].items():
            if 'customvalue' in v:
                for tv in v['customvalue']:
                    newkey = None
                    field_name = self.custom_fields[tv['key']] if tv['key'] in self.custom_fields else tv['key']
                    if field_name in self.groupby_custom_field_excludes:
                        continue
                    values = []
                    keylist = map(lambda x: x.strip(), tv['value'].split(','))
                    for kl in keylist:
                        try:
                            newkey = '%s%s_%s' % (self.config.get('vmware', 'custom_field_group_prefix'), str(field_name), kl)
                            newkey = newkey.strip()
                        except Exception as e:
                            self.debugl(e)
                        values.append(newkey)
                    for tag in values:
                        if not tag:
                            continue
                        if tag not in inventory:
                            inventory[tag] = {}
                            inventory[tag]['hosts'] = []
                        if k not in inventory[tag]['hosts']:
                            inventory[tag]['hosts'].append(k)
    return inventory