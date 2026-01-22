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
def test_smart_request_content_type_with_directive_check(self):
    from urllib3.response import HTTPResponse

    class PoolManagerMock:

        def __init__(self) -> None:
            self.headers: Dict[str, str] = {}

        def request(self, method, url, fields=None, headers=None, redirect=True, preload_content=True):
            return HTTPResponse(headers={'Content-Type': 'application/x-git-upload-pack-result; charset=utf-8'}, request_method=method, request_url=url, preload_content=preload_content, status=200)
    clone_url = 'https://hacktivis.me/git/blog.git/'
    client = HttpGitClient(clone_url, pool_manager=PoolManagerMock(), config=None)
    self.assertTrue(client._smart_request('git-upload-pack', clone_url, data=None))