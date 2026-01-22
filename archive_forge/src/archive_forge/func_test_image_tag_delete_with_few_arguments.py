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
def test_image_tag_delete_with_few_arguments(self):
    args = self._make_args({'image_id': 'IMG-01', 'tag_value': None})
    msg = 'Unable to delete tag. Specify image_id and tag_value'
    self.assert_exits_with_msg(func=test_shell.do_image_tag_delete, func_args=args, err_msg=msg)