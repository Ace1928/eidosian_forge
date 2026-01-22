from __future__ import absolute_import, unicode_literals
import typing
import contextlib
import io
import os
import six
import time
from collections import OrderedDict
from threading import RLock
from . import errors
from ._typing import overload
from .base import FS
from .copy import copy_modified_time
from .enums import ResourceType, Seek
from .info import Info
from .mode import Mode
from .path import iteratepath, normpath, split
def to_info(self, namespaces=None):
    namespaces = namespaces or ()
    info = {'basic': {'name': self.name, 'is_dir': self.is_dir}}
    if 'details' in namespaces:
        info['details'] = {'_write': ['accessed', 'modified'], 'type': int(self.resource_type), 'size': self.size, 'accessed': self.accessed_time, 'modified': self.modified_time, 'created': self.created_time}
    return Info(info)