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
@utils.exclusive_arg('detach_network_port', '--network', metavar='<network>', help='The neutron network that container will detach from.')
@utils.exclusive_arg('detach_network_port', '--port', metavar='<port>', help='The neutron port that container will detach from.')
@utils.arg('container', metavar='<container>', help='ID or name of the container to detach the network.')
def do_network_detach(cs, args):
    """Detach a network from the container."""
    opts = {}
    opts['container'] = args.container
    opts['network'] = args.network
    opts['port'] = args.port
    opts = zun_utils.remove_null_parms(**opts)
    try:
        cs.containers.network_detach(**opts)
        print('Request to detach network from container %s has been accepted.' % args.container)
    except Exception as e:
        print('Detach network from container %(container)s failed: %(e)s' % {'container': args.container, 'e': e})