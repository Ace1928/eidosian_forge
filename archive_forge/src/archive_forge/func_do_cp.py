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
@utils.arg('source', metavar='<source>', help='The source should be copied to the container or localhost. The format of this parameter is [container:]src_path.')
@utils.arg('destination', metavar='<destination>', help='The directory destination where save the source. The format of this parameter is [container:]dest_path.')
def do_cp(cs, args):
    """Copy files/tars between a container and the local filesystem."""
    if ':' in args.source:
        source_parts = args.source.split(':', 1)
        container_id = source_parts[0]
        container_path = source_parts[1]
        opts = {}
        opts['id'] = container_id
        opts['path'] = container_path
        res = cs.containers.get_archive(**opts)
        dest_path = args.destination
        tardata = io.BytesIO(res['data'])
        with closing(tarfile.open(fileobj=tardata)) as tar:
            tar.extractall(dest_path)
    elif ':' in args.destination:
        dest_parts = args.destination.split(':', 1)
        container_id = dest_parts[0]
        container_path = dest_parts[1]
        filename = os.path.split(args.source)[1]
        opts = {}
        opts['id'] = container_id
        opts['path'] = container_path
        tardata = io.BytesIO()
        with closing(tarfile.open(fileobj=tardata, mode='w')) as tar:
            tar.add(args.source, arcname=filename)
        opts['data'] = tardata.getvalue()
        cs.containers.put_archive(**opts)
    else:
        print('Please check the parameters for zun copy!')
        print('Usage:')
        print('zun cp container:src_path dest_path|-')
        print('zun cp src_path|- container:dest_path')