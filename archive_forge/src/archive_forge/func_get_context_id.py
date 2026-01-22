import os
import json
import hashlib
from docker import utils
from docker.constants import IS_WINDOWS_PLATFORM
from docker.constants import DEFAULT_UNIX_SOCKET
from docker.utils.config import find_config_file
def get_context_id(name):
    return hashlib.sha256(name.encode('utf-8')).hexdigest()