import argparse
from copy import deepcopy
import io
import json
import os
from unittest import mock
import sys
import tempfile
import testtools
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell
from glanceclient.v2 import shell as test_shell  # noqa
def test_do_cache_list_endpoint_not_provided(self):
    args = self._make_args({})
    self.gc.endpoint_provided = False
    with mock.patch('glanceclient.common.utils.exit') as mock_exit:
        test_shell.do_cache_list(self.gc, args)
        mock_exit.assert_called_once_with('Direct server endpoint needs to be provided. Do not use loadbalanced or catalog endpoints.')