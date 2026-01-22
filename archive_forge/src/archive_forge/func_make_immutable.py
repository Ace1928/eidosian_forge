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
def make_immutable(value, strict=True):
    value_type = type(value)
    if value_type is dict:
        return ImmutableDict(value)
    if value_type is list:
        return ImmutableList(value)
    if strict:
        if value_type not in _json_compatible_types:
            raise TypeError("Type {} can't be immutable.".format(value_type))
    return value