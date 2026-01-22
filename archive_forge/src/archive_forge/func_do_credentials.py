from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
def do_credentials(cs, args):
    """Show user credentials returned from auth."""
    catalog = cs.keystone_client.service_catalog.catalog
    cliutils.print_dict(catalog['user'], 'User Credentials')
    if not catalog['version'] == 'v3':
        data = catalog['token']
    else:
        data = {'issued_at': catalog['issued_at'], 'expires': catalog['expires_at'], 'id': catalog['auth_token'], 'audit_ids': catalog['audit_ids'], 'tenant': catalog['project']}
    cliutils.print_dict(data, 'Token')