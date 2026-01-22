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
@utils.arg('containers', metavar='<container>', nargs='+', help='ID of the (container)s to rebuild.')
@utils.arg('--image', metavar='<image>', help='The image for specified container to update.')
@utils.arg('--image-driver', metavar='<image_driver>', help='The image driver to use to pull container image. It can have following values: "docker": pull the image from Docker Hub. "glance": pull the image from Glance. The default value is source container\'s image driver ')
def do_rebuild(cs, args):
    """Rebuild specified containers."""
    for container in args.containers:
        opts = {}
        opts['id'] = container
        if args.image:
            opts['image'] = args.image
        if args.image_driver:
            opts['image_driver'] = args.image_driver
        try:
            cs.containers.rebuild(**opts)
            print('Request to rebuild container %s has been accepted.' % container)
        except Exception as e:
            print('Rebuild for container %(container)s failed: %(e)s' % {'container': container, 'e': e})