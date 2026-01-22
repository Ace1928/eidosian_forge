import argparse
import getpass
import io
import json
import logging
import os
from cliff import columns as cliff_columns
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common as network_common
def bool_from_str(value, strict=False):
    true_strings = ('1', 't', 'true', 'on', 'y', 'yes')
    false_strings = ('0', 'f', 'false', 'off', 'n', 'no')
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in true_strings:
        return True
    elif lowered in false_strings or not strict:
        return False
    msg = _("Unrecognized value '%(value)s'; acceptable values are: %(valid)s") % {'value': value, 'valid': ', '.join((f"'{s}'" for s in sorted(true_strings + false_strings)))}
    raise ValueError(msg)