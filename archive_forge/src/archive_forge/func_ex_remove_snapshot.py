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
def ex_remove_snapshot(self, node, snapshot_name=None, remove_children=True):
    """
        Remove a snapshot from node.
        If snapshot_name is not defined remove the last one.
        """
    if self.driver_soap is None:
        self._get_soap_driver()
    return self.driver_soap.ex_remove_snapshot(node, snapshot_name=snapshot_name, remove_children=remove_children)