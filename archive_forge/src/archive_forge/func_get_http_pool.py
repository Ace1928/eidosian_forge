import hashlib
import io
import json
import os
import platform
import random
import socket
import ssl
import threading
import time
import urllib.parse
from typing import (
import filelock
import urllib3
from blobfile import _xml as xml
def get_http_pool(self) -> urllib3.PoolManager:
    if self._get_http_pool is None:
        return global_pool_director.get_http_pool()
    else:
        return self._get_http_pool()