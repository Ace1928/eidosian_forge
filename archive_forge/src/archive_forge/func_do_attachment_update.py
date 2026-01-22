import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
@api_versions.wraps('3.27')
@utils.arg('attachment', metavar='<attachment>', help='ID of attachment.')
@utils.arg('--initiator', metavar='<initiator>', default=None, help='iqn of the initiator attaching to.  Default=None.')
@utils.arg('--ip', metavar='<ip>', default=None, help='ip of the system attaching to.  Default=None.')
@utils.arg('--host', metavar='<host>', default=None, help='Name of the host attaching to. Default=None.')
@utils.arg('--platform', metavar='<platform>', default='x86_64', help='Platform type. Default=x86_64.')
@utils.arg('--ostype', metavar='<ostype>', default='linux2', help='OS type. Default=linux2.')
@utils.arg('--multipath', metavar='<multipath>', default=False, help='Use multipath. Default=False.')
@utils.arg('--mountpoint', metavar='<mountpoint>', default=None, help='Mountpoint volume will be attached at. Default=None.')
def do_attachment_update(cs, args):
    """Update an attachment for a cinder volume.
    This call is designed to be more of an attachment completion than anything
    else.  It expects the value of a connector object to notify the driver that
    the volume is going to be connected and where it's being connected to.
    """
    connector = {'initiator': args.initiator, 'ip': args.ip, 'platform': args.platform, 'host': args.host, 'os_type': args.ostype, 'multipath': args.multipath, 'mountpoint': args.mountpoint}
    attachment = cs.attachments.update(args.attachment, connector)
    attachment_dict = attachment.to_dict()
    connector_dict = attachment_dict.pop('connection_info', None)
    shell_utils.print_dict(attachment_dict)
    if connector_dict:
        shell_utils.print_dict(connector_dict)