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
def test_do_md_namespace_import_no_input(self):
    args = self._make_args({'file': None})
    utils.get_data_file = mock.Mock(return_value=None)
    self.assertRaises(SystemExit, test_shell.do_md_namespace_import, self.gc, args)