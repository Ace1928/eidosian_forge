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
def test_fuzzy_same_distance(self):
    cmd_mgr = commandmanager.CommandManager('cliff.fuzzy')
    app = application.App('test', '1.0', cmd_mgr)
    cmd_mgr.add_command('user', test_utils.TestCommand)
    for cmd in cmd_mgr.commands.keys():
        self.assertEqual(8, utils.damerau_levenshtein('node', cmd, utils.COST))
    matches = app.get_fuzzy_matches('node')
    self.assertEqual(['complete', 'help', 'user'], matches)