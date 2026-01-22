import argparse
from contextlib import closing
import io
import os
import tarfile
import time
import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.common.websocketclient import websocketclient
from zunclient import exceptions as exc
@utils.deprecated(SG_DEPRECATION_MESSAGE)
@utils.arg('container', metavar='<container>', help='ID or name of the container to add security group.')
@utils.arg('security_group', metavar='<security_group>', help='Security group ID or name for specified container.')
def do_add_security_group(cs, args):
    """Add security group for specified container."""
    opts = {}
    opts['id'] = args.container
    opts['security_group'] = args.security_group
    opts = zun_utils.remove_null_parms(**opts)
    try:
        cs.containers.add_security_group(**opts)
        print('Request to add security group for container %s has been accepted.' % args.container)
    except Exception as e:
        print('Add security group for container %(container)s failed: %(e)s' % {'container': args.container, 'e': e})