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
def test_do_stores_info_with_detail(self):
    args = self._make_args({'detail': True})
    with mock.patch.object(self.gc.images, 'get_stores_info_detail') as mocked_list:
        mocked_list.return_value = self.stores_info_detail_response
        test_shell.do_stores_info(self.gc, args)
        mocked_list.assert_called_once_with()
        utils.print_dict.assert_called_once_with(self.stores_info_detail_response)