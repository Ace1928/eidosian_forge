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
def test_do_image_delete_deleted(self):
    image_id = 'deleted-img'
    args = argparse.Namespace(id=[image_id])
    with mock.patch.object(self.gc.images, 'delete') as mocked_delete:
        mocked_delete.side_effect = exc.HTTPNotFound
        self.assert_exits_with_msg(func=test_shell.do_image_delete, func_args=args)