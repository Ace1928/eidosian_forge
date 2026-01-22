from __future__ import absolute_import, division, print_function
import ast
import base64
import json
import os
import re
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.connection import ConnectionError
from ansible.plugins.httpapi import HttpApiBase
from copy import copy, deepcopy
def set_hosts(self):
    if self.params.get('host') is not None:
        hosts = ast.literal_eval(self.params.get('host')) if '[' in self.params.get('host') else self.params.get('host').split(',')
    else:
        if self.inventory_hosts is None:
            self.inventory_hosts = re.sub('[[\\]]', '', self.connection.get_option('host')).split(',')
        hosts = self.inventory_hosts
    if self.provided_hosts is None:
        self.provided_hosts = deepcopy(hosts)
        self.connection.queue_message('debug', 'Provided Hosts: {0}'.format(self.provided_hosts))
        self.backup_hosts = deepcopy(hosts)
        self.current_host = self.backup_hosts.pop(0)
        self.connection.queue_message('debug', 'Initializing operation on {0}'.format(self.current_host))
    elif self.provided_hosts != hosts:
        self.provided_hosts = deepcopy(hosts)
        self.connection.queue_message('debug', 'Provided Hosts have changed: {0}'.format(self.provided_hosts))
        self.backup_hosts = deepcopy(hosts)
        try:
            self.backup_hosts.pop(self.backup_hosts.index(self.current_host))
            self.connection.queue_message('debug', 'Connected host {0} found in the provided hosts. Continuing with it.'.format(self.current_host))
        except Exception:
            self.current_host = self.backup_hosts.pop(0)
            self.connection._connected = False
            self.connection.queue_message('debug', 'Initializing operation on {0}'.format(self.current_host))
    self.connection.set_option('host', self.current_host)