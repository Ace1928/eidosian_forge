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
def test_url_redirect_location(self):
    from urllib3.response import HTTPResponse
    test_data = {'https://gitlab.com/inkscape/inkscape/': {'location': 'https://gitlab.com/inkscape/inkscape.git/', 'redirect_url': 'https://gitlab.com/inkscape/inkscape.git/', 'refs_data': b'001e# service=git-upload-pack\n00000032fb2bebf4919a011f0fd7cec085443d0031228e76 HEAD\n0000'}, 'https://github.com/jelmer/dulwich/': {'location': 'https://github.com/jelmer/dulwich/', 'redirect_url': 'https://github.com/jelmer/dulwich/', 'refs_data': b'001e# service=git-upload-pack\n000000323ff25e09724aa4d86ea5bca7d5dd0399a3c8bfcf HEAD\n0000'}, 'https://codeberg.org/ashwinvis/radicale-sh.git/': {'location': '/ashwinvis/radicale-auth-sh/', 'redirect_url': 'https://codeberg.org/ashwinvis/radicale-auth-sh/', 'refs_data': b'001e# service=git-upload-pack\n00000032470f8603768b608fc988675de2fae8f963c21158 HEAD\n0000'}}
    tail = 'info/refs?service=git-upload-pack'

    class PoolManagerMock:

        def __init__(self) -> None:
            self.headers: Dict[str, str] = {}

        def request(self, method, url, fields=None, headers=None, redirect=True, preload_content=True):
            base_url = url[:-len(tail)]
            redirect_base_url = test_data[base_url]['location']
            redirect_url = redirect_base_url + tail
            headers = {'Content-Type': 'application/x-git-upload-pack-advertisement'}
            body = test_data[base_url]['refs_data']
            status = 200
            request_url = redirect_url
            if redirect is False:
                request_url = url
                if redirect_base_url != base_url:
                    body = b''
                    headers['location'] = test_data[base_url]['location']
                    status = 301
            return HTTPResponse(body=BytesIO(body), headers=headers, request_method=method, request_url=request_url, preload_content=preload_content, status=status)
    pool_manager = PoolManagerMock()
    for base_url in test_data.keys():
        c = HttpGitClient(base_url, pool_manager=pool_manager, config=None)
        _, _, processed_url = c._discover_references(b'git-upload-pack', base_url)
        resp = c.pool_manager.request('GET', base_url + tail, redirect=False)
        redirect_location = resp.get_redirect_location()
        if resp.status == 200:
            self.assertFalse(redirect_location)
        if redirect_location:
            self.assertEqual(processed_url, test_data[base_url]['redirect_url'])
        else:
            self.assertEqual(processed_url, base_url)