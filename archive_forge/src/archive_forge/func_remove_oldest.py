import enum
import os
import socket
import subprocess
import uuid
from platform import uname
from typing import List, Tuple, Union
from packaging.version import parse, Version
import psutil
import torch
import asyncio
from functools import partial
from typing import (
from collections import OrderedDict
from typing import Any, Hashable, Optional
from vllm.logger import init_logger
def remove_oldest(self):
    if not self.cache:
        return
    key, value = self.cache.popitem(last=False)
    self._on_remove(key, value)