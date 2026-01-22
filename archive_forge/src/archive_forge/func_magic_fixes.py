import argparse as argparse_mod
import collections
import copy
import errno
import json
import os
import re
import sys
import typing as ty
import warnings
from keystoneauth1 import adapter
from keystoneauth1 import loading
import platformdirs
import yaml
from openstack import _log
from openstack.config import _util
from openstack.config import cloud_region
from openstack.config import defaults
from openstack.config import vendors
from openstack import exceptions
from openstack import warnings as os_warnings
def magic_fixes(self, config):
    """Perform the set of magic argument fixups"""
    if 'auth' in config and 'token' in config['auth'] or ('auth_token' in config and config['auth_token']) or ('token' in config and config['token']):
        config.setdefault('token', config.pop('auth_token', None))
    if 'auth' in config and 'passcode' in config:
        config['auth']['passcode'] = config.pop('passcode', None)
    config = self._fix_backwards_api_timeout(config)
    if 'endpoint_type' in config:
        config['interface'] = config.pop('endpoint_type')
    config = self._fix_backwards_auth_plugin(config)
    config = self._fix_backwards_project(config)
    config = self._fix_backwards_interface(config)
    config = self._fix_backwards_networks(config)
    config = self._handle_domain_id(config)
    for key in BOOL_KEYS:
        if key in config:
            if type(config[key]) is not bool:
                config[key] = get_boolean(config[key])
    for key in CSV_KEYS:
        if key in config:
            if isinstance(config[key], str):
                config[key] = config[key].split(',')
    if 'auth' in config and 'auth_url' in config['auth']:
        config['auth']['auth_url'] = config['auth']['auth_url'].format(**config)
    return config