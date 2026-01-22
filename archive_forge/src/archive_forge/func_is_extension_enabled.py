import json
import logging
import os
import shlex
import subprocess
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
import testtools
@classmethod
def is_extension_enabled(cls, alias, *, service='network'):
    """Ask client cloud if extension is enabled"""
    extensions = cls.openstack(f'extension list --{service}', parse_output=True)
    return alias in [x['Alias'] for x in extensions]