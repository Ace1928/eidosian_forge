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
def linode_set_datacenter(self, dc):
    """
        Set the default datacenter for Linode creation

        Since Linodes must be created in a facility, this function sets the
        default that :class:`create_node` will use.  If a location keyword is
        not passed to :class:`create_node`, this method must have already been
        used.

        :keyword dc: the datacenter to create Linodes in unless specified
        :type    dc: :class:`NodeLocation`

        :rtype: ``bool``
        """
    did = dc.id
    params = {'api_action': 'avail.datacenters'}
    data = self.connection.request(API_ROOT, params=params).objects[0]
    for datacenter in data:
        if did == dc['DATACENTERID']:
            self.datacenter = did
            return
    dcs = ', '.join([d['DATACENTERID'] for d in data])
    self.datacenter = None
    raise LinodeException(253, 'Invalid datacenter (use one of %s)' % dcs)