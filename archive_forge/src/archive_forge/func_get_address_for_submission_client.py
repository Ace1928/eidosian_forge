import abc
import asyncio
import datetime
import functools
import importlib
import json
import logging
import os
import pkgutil
from abc import ABCMeta, abstractmethod
from base64 import b64decode
from collections import namedtuple
from collections.abc import MutableMapping, Mapping, Sequence
from typing import Optional
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services as services
from ray._raylet import GcsClient
from ray._private.utils import split_address
import aiosignal  # noqa: F401
import ray._private.protobuf_compat
from frozenlist import FrozenList  # noqa: F401
from ray._private.utils import binary_to_hex, check_dashboard_dependencies_installed
def get_address_for_submission_client(address: Optional[str]) -> str:
    """Get Ray API server address from Ray bootstrap or Client address.

    If None, it will try to auto-detect a running Ray instance, or look
    for local GCS process.

    `address` is always overridden by the RAY_ADDRESS environment
    variable, just like the `address` argument in `ray.init()`.

    Args:
        address: Ray cluster bootstrap address or Ray Client address.
            Could also be "auto".

    Returns:
        API server HTTP URL, e.g. "http://<head-node-ip>:8265".
    """
    if os.environ.get('RAY_ADDRESS'):
        logger.debug(f'Using RAY_ADDRESS={os.environ['RAY_ADDRESS']}')
        address = os.environ['RAY_ADDRESS']
    if address and '://' in address:
        module_string, _ = split_address(address)
        if module_string == 'ray':
            logger.debug(f'Retrieving API server address from Ray Client address {address}...')
            address = ray_client_address_to_api_server_url(address)
    else:
        address = ray_address_to_api_server_url(address)
    logger.debug(f'Using API server address {address}.')
    return address