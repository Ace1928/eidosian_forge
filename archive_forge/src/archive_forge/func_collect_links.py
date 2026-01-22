import json
import tempfile
from unittest import mock
import io
from oslo_serialization import base64
import testtools
from testtools import matchers
from urllib import error
import yaml
from heatclient.common import template_utils
from heatclient.common import utils
from heatclient import exc
def collect_links(self, env, content, url, env_base_url=''):
    jenv = yaml.safe_load(env)
    files = {}
    if url:

        def side_effect(args):
            if url == args:
                return io.BytesIO(content)
        with mock.patch('urllib.request.urlopen') as mock_url:
            mock_url.side_effect = side_effect
            template_utils.resolve_environment_urls(jenv.get('resource_registry'), files, env_base_url)
            self.assertEqual(content.decode('utf-8'), files[url])
    else:
        template_utils.resolve_environment_urls(jenv.get('resource_registry'), files, env_base_url)