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
@utils.arg('container', metavar='<container>', help='ID or name of the container to get logs for.')
@utils.arg('--stdout', action='store_true', help='Only stdout logs of container.')
@utils.arg('--stderr', action='store_true', help='Only stderr logs of container.')
@utils.arg('--since', metavar='<since>', default=None, help='Show logs since a given datetime or integer epoch (in seconds).')
@utils.arg('-t', '--timestamps', dest='timestamps', action='store_true', default=False, help='Show timestamps.')
@utils.arg('--tail', metavar='<tail>', default='all', help='Number of lines to show from the end of the logs.')
def do_logs(cs, args):
    """Get logs of a container."""
    opts = {}
    opts['id'] = args.container
    opts['stdout'] = args.stdout
    opts['stderr'] = args.stderr
    opts['since'] = args.since
    opts['timestamps'] = args.timestamps
    opts['tail'] = args.tail
    opts = zun_utils.remove_null_parms(**opts)
    logs = cs.containers.logs(**opts)
    print(logs)