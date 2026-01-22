import asyncio
import enum
import json
import pathlib
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import (
from traitlets import Any as Any_
from traitlets import Instance
from traitlets import List as List_
from traitlets import Unicode, default
from traitlets.config import LoggingConfigurable
@classmethod
def unregister_message_listener(cls, listener: 'HandlerListenerCallback'):
    """unregister a listener for language server protocol messages"""
    for scope in MessageScope:
        cls._listeners[str(scope.value)] = [lst for lst in cls._listeners[str(scope.value)] if lst.listener != listener]