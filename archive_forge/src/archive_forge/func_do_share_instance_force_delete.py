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
@cliutils.arg('instance', metavar='<instance>', nargs='+', help='Name or ID of the instance(s) to force delete.')
@api_versions.wraps('2.3')
@cliutils.arg('--wait', action='store_true', help='Wait for share instance deletion')
@cliutils.service_type('sharev2')
def do_share_instance_force_delete(cs, args):
    """Force-delete the share instance, regardless of state (Admin only)."""
    failure_count = 0
    instances_to_delete = []
    for instance in args.instance:
        try:
            instance_ref = _find_share_instance(cs, instance)
            instances_to_delete.append(instance_ref)
            instance_ref.force_delete()
        except Exception as e:
            failure_count += 1
            print('Delete for share instance %s failed: %s' % (instance, e), file=sys.stderr)
    if failure_count == len(args.instance):
        raise exceptions.CommandError('Unable to force delete any of specified share instances.')
    if args.wait:
        for instance in instances_to_delete:
            try:
                _wait_for_resource_status(cs, instance, resource_type='share_instance', expected_status='deleted')
            except exceptions.CommandError as e:
                print(e, file=sys.stderr)