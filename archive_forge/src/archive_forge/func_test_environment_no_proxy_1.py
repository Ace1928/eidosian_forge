import base64
import os
import shutil
import sys
import tempfile
import warnings
from io import BytesIO
from typing import Dict
from unittest.mock import patch
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
import dulwich
from dulwich import client
from dulwich.tests import TestCase, skipIf
from ..client import (
from ..config import ConfigDict
from ..objects import Commit, Tree
from ..pack import pack_objects_to_data, write_pack_data, write_pack_objects
from ..protocol import TCP_GIT_PORT, Protocol
from ..repo import MemoryRepo, Repo
from .utils import open_repo, setup_warning_catcher, tear_down_repo
def test_environment_no_proxy_1(self):
    import urllib3
    config = ConfigDict()
    self.overrideEnv('http_proxy', 'http://myproxy:8080')
    self.overrideEnv('no_proxy', 'xyz,abc.def.gh,abc.gh')
    base_url = 'http://xyz.abc.def.gh:8080/path/port'
    manager = default_urllib3_manager(config=config, base_url=base_url)
    self.assertNotIsInstance(manager, urllib3.ProxyManager)
    self.assertIsInstance(manager, urllib3.PoolManager)