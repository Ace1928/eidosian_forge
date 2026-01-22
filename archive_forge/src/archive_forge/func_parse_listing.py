import json
import logging
import os
import shlex
import subprocess
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
import testtools
def parse_listing(self, raw_output):
    """Return list of dicts with basic item parsed from cli output."""
    return output_parser.listing(raw_output)