import unittest
from charset_normalizer.cli.normalizer import cli_detect, query_yes_no
from unittest.mock import patch
from os.path import exists
from os import remove
def test_non_existent_file(self):
    with self.assertRaises(SystemExit) as cm:
        cli_detect(['./data/not_found_data.txt'])
    self.assertEqual(cm.exception.code, 2)