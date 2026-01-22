import unittest
from charset_normalizer.cli.normalizer import cli_detect, query_yes_no
from unittest.mock import patch
from os.path import exists
from os import remove
def test_single_verbose_file(self):
    self.assertEqual(0, cli_detect(['./data/sample-arabic-1.txt', '--verbose']))