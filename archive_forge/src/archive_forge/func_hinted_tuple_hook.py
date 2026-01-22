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
def hinted_tuple_hook(obj: Any) -> Any:
    if '__tuple__' in obj:
        return tuple(obj['items'])
    else:
        return obj