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
def test_config_proxy_creds(self):
    import urllib3
    config = ConfigDict()
    config.set(b'http', b'proxy', b'http://jelmer:example@localhost:3128/')
    manager = default_urllib3_manager(config=config)
    assert isinstance(manager, urllib3.ProxyManager)
    self.assertEqual(manager.proxy_headers, {'proxy-authorization': 'Basic amVsbWVyOmV4YW1wbGU='})