import argparse
import getpass
import logging
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from zunclient import api_versions
from zunclient import client as base_client
from zunclient.common.apiclient import auth
from zunclient.common import cliutils
from zunclient import exceptions as exc
from zunclient.i18n import _
from zunclient.v1 import shell as shell_v1
from zunclient import version
@property
def management_url(self):
    if not HAS_KEYRING or not self.args.os_cache:
        return None
    management_url = None
    try:
        block = keyring.get_password('zunclient_auth', self._make_key())
        if block:
            _token, management_url, _tenant_id = block.split('|', 2)
    except all_errors:
        pass
    return management_url