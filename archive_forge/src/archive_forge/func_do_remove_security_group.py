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
@utils.arg('container', metavar='<container>', help='ID or name of the container to remove security group.')
@utils.arg('security_group', metavar='<security_group>', help='The security group to remove from specified container.')
def do_remove_security_group(cs, args):
    """Remove security group for specified container."""
    opts = {}
    opts['id'] = args.container
    opts['security_group'] = args.security_group
    opts = zun_utils.remove_null_parms(**opts)
    try:
        cs.containers.remove_security_group(**opts)
        print('Request to remove security group for container %s has been accepted.' % args.container)
    except Exception as e:
        print('Remove security group for container %(container)s failed: %(e)s' % {'container': args.container, 'e': e})