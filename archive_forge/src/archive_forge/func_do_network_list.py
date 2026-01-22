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
@utils.arg('container', metavar='<container>', help='ID or name of the container to display network info.')
def do_network_list(cs, args):
    """List networks on a container"""
    opts = {}
    opts['container'] = args.container
    opts = zun_utils.remove_null_parms(**opts)
    networks = cs.containers.network_list(**opts)
    zun_utils.list_container_networks(networks)