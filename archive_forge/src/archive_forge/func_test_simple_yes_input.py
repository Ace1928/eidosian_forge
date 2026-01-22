import unittest
from charset_normalizer.cli.normalizer import cli_detect, query_yes_no
from unittest.mock import patch
from os.path import exists
from os import remove
@patch('builtins.input', lambda *args: 'y')
def test_simple_yes_input(self):
    self.assertTrue(query_yes_no('Are u willing to chill a little bit ?'))