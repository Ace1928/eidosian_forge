import ssl
import json
import time
import atexit
import base64
import asyncio
import hashlib
import logging
import warnings
import functools
import itertools
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def find_by_uuid(self, node_uuid):
    """Searches VMs for a given uuid
        returns pyVmomi.VmomiSupport.vim.VirtualMachine
        """
    vm = self.connection.content.searchIndex.FindByUuid(None, node_uuid, True, True)
    if not vm:
        vm = self._get_item_by_moid('VirtualMachine', node_uuid)
        if not vm:
            raise LibcloudError('Unable to locate VirtualMachine.', driver=self)
    return vm