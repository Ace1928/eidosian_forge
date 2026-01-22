import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def test_check_for_exceptions(self):
    status = [400, 422, 500]
    for s in status:
        resp = mock.Mock()
        resp.status = s
        self.assertRaises(Exception, common.check_for_exceptions, resp, 'body')
    resp = mock.Mock()
    resp.status_code = 200
    common.check_for_exceptions(resp, 'body')