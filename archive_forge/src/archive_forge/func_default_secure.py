from __future__ import annotations
import hashlib
import hmac
import json
import logging
import os
import pickle
import pprint
import random
import typing as t
import warnings
from binascii import b2a_hex
from datetime import datetime, timezone
from hmac import compare_digest
import zmq.asyncio
from tornado.ioloop import IOLoop
from traitlets import (
from traitlets.config.configurable import Configurable, LoggingConfigurable
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
from zmq.eventloop.zmqstream import ZMQStream
from ._version import protocol_version
from .adapter import adapt
from .jsonutil import extract_dates, json_clean, json_default, squash_dates
def default_secure(cfg: t.Any) -> None:
    """Set the default behavior for a config environment to be secure.

    If Session.key/keyfile have not been set, set Session.key to
    a new random UUID.
    """
    warnings.warn('default_secure is deprecated', DeprecationWarning, stacklevel=2)
    if 'Session' in cfg and ('key' in cfg.Session or 'keyfile' in cfg.Session):
        return
    cfg.Session.key = new_id_bytes()