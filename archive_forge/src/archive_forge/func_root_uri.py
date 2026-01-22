from __future__ import annotations
import os
import json
import socket
import contextlib
import logging
from typing import Optional, Dict, Any, Union, Type, Mapping, Callable, List
from lazyops.utils.logs import default_logger as logger
import aiokeydb.v2.exceptions as exceptions
from aiokeydb.v2.types import BaseSettings, validator, root_validator, lazyproperty, KeyDBUri
from aiokeydb.v2.serializers import SerializerType, BaseSerializer
from aiokeydb.v2.utils import import_string
from aiokeydb.v2.configs.worker import KeyDBWorkerSettings
from aiokeydb.v2.backoff import default_backoff
@property
def root_uri(self) -> str:
    """
        Returns the Root URI
        -> host:port
        """
    if not self.url:
        return f'{self.host}:{self.port}'
    base = self.url
    if '/' in base[-4:]:
        base = base.rsplit('/', 1)[0]
    if '://' in base:
        base = base.split('://', 1)[-1]
    return base