import base64
import binascii
import os
import re
import shlex
from oslo_serialization import jsonutils
from oslo_utils import netutils
from urllib import parse
from urllib import request
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import cliutils as utils
from zunclient import exceptions as exc
from zunclient.i18n import _
def parse_mounts(mounts):
    err_msg = "Invalid mounts argument '%s'. mounts arguments must be of the form --mount source=<volume>,destination=<path>, or use --mount size=<size>,destination=<path> to create a new volume and mount to the container, or use --mount type=bind,source=<file>,destination=<path> to inject file into a path in the container."
    parsed_mounts = []
    for mount in mounts:
        keys = ['source', 'destination', 'size', 'type']
        mount_info = {}
        for mnt in mount.split(','):
            try:
                k, v = mnt.split('=', 1)
                k = k.strip()
                v = v.strip()
            except ValueError:
                raise apiexec.CommandError(err_msg % mnt)
            if k in keys:
                if mount_info.get(k):
                    raise apiexec.CommandError(err_msg % mnt)
                mount_info[k] = v
            else:
                raise apiexec.CommandError(err_msg % mnt)
        if not mount_info.get('destination'):
            raise apiexec.CommandError(err_msg % mount)
        if not mount_info.get('source') and (not mount_info.get('size')):
            raise apiexec.CommandError(err_msg % mount)
        type = mount_info.get('type', 'volume')
        if type not in ('volume', 'bind'):
            mnt = 'type=%s' % type
            raise apiexec.CommandError(err_msg % mnt)
        if type == 'bind':
            filename = mount_info.pop('source')
            with open(filename, 'rb') as file:
                mount_info['source'] = file.read()
        parsed_mounts.append(mount_info)
    return parsed_mounts