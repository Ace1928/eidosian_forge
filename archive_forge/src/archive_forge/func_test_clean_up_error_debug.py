import argparse
import codecs
import io
from unittest import mock
from cliff import app as application
from cliff import command as c_cmd
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils as test_utils
from cliff import utils
import sys
def test_clean_up_error_debug(self):
    app, command = make_app()
    app.clean_up = mock.MagicMock(name='clean_up')
    ret = app.run(['--debug', 'error'])
    self.assertNotEqual(ret, 0)
    self.assertTrue(app.clean_up.called)
    call_args = app.clean_up.call_args_list[0]
    self.assertEqual(mock.call(mock.ANY, 1, mock.ANY), call_args)
    args, kwargs = call_args
    self.assertIsInstance(args[2], RuntimeError)
    self.assertEqual(('test exception',), args[2].args)