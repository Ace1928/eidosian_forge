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
def test_normal_clean_up_raises_exception_debug(self):
    app, command = make_app()
    app.clean_up = mock.MagicMock(name='clean_up', side_effect=RuntimeError('within clean_up'))
    app.run(['--debug', 'mock'])
    self.assertTrue(app.clean_up.called)
    call_args = app.clean_up.call_args_list[0]
    self.assertEqual(mock.call(mock.ANY, 0, None), call_args)