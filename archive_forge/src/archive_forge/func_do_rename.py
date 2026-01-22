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
@utils.deprecated(RENAME_DEPRECATION_MESSAGE)
@utils.arg('container', metavar='<container>', help='ID or name of the container to rename.')
@utils.arg('name', metavar='<name>', help='The new name for the container')
def do_rename(cs, args):
    """Rename a container."""
    cs.containers.rename(args.container, args.name)