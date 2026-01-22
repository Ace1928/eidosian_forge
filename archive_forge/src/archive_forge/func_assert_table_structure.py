import json
import logging
import os
import shlex
import subprocess
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
import testtools
def assert_table_structure(self, items, field_names):
    """Verify that all items have keys listed in field_names."""
    for item in items:
        for field in field_names:
            self.assertIn(field, item)