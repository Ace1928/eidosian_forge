import os
import re
import binascii
import itertools
from copy import copy
from datetime import datetime
from libcloud.utils.py3 import httplib
from libcloud.compute.base import (
from libcloud.common.linode import (
from libcloud.compute.types import Provider, NodeState, StorageVolumeState
from libcloud.utils.networking import is_private_subnet
def ex_create_disk(self, size, name, node, fs_type, image=None, ex_root_pass=None, ex_authorized_keys=None, ex_authorized_users=None, ex_read_only=False):
    """
        Adds a new disk to node

        :param    size: Size of disk in megabytes (required)
        :type       size: ``int``

        :param    name: Name of the disk to be created (required)
        :type       name: ``str``

        :param    node: Node to attach disk to (required)
        :type       node: :class:`Node`

        :param    fs_type: The formatted type of this disk. Valid types are:
                             ext3, ext4, swap, raw, initrd
        :type       fs_type: ``str``

        :keyword    image: Image  to deploy the volume from
        :type       image: :class:`NodeImage`

        :keyword    ex_root_pass: root password,required                     if an image is provided
        :type       ex_root_pass: ``str``

        :keyword ex_authorized_keys:  a list of SSH keys
        :type    ex_authorized_keys: ``list`` of ``str``

        :keyword ex_authorized_users:  a list of usernames                  that will have their SSH keys,                 if any, automatically appended                  to the root user's ~/.ssh/authorized_keys file.
        :type    ex_authorized_users: ``list`` of ``str``

        :keyword ex_read_only: if true, this disk is read-only
        :type ex_read_only: ``bool``

        :return: LinodeDisk representing the newly-created disk
        :rtype: :class:`LinodeDisk`
        """
    attr = {'label': str(name), 'size': int(size), 'filesystem': fs_type, 'read_only': ex_read_only}
    if not isinstance(node, Node):
        raise LinodeExceptionV4('Invalid node instance')
    if fs_type not in self._linode_disk_filesystems:
        raise LinodeExceptionV4('Not valid filesystem type')
    if image is not None:
        if not isinstance(image, NodeImage):
            raise LinodeExceptionV4('Invalid image instance')
        if ex_root_pass is None:
            raise LinodeExceptionV4('root_pass is required when deploying an image')
        attr['image'] = image.id
        attr['root_pass'] = ex_root_pass
    if ex_authorized_keys is not None:
        attr['authorized_keys'] = list(ex_authorized_keys)
    if ex_authorized_users is not None:
        attr['authorized_users'] = list(ex_authorized_users)
    response = self.connection.request('/v4/linode/instances/%s/disks' % node.id, data=json.dumps(attr), method='POST').object
    return self._to_disk(response)