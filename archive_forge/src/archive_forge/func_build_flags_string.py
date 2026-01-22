import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def build_flags_string(flags):
    """Format a string of value-less flags.

    Pass in a dictionary mapping flags to booleans. Those flags set to true
    are included in the returned string.

    Usage::

        build_flags_string({
            '--no-ttl': True,
            '--no-name': False,
            '--verbose': True,
        })

    Returns::

        '--no-ttl --verbose'
    """
    flags = {flag: is_set for flag, is_set in flags.items() if is_set}
    return ' '.join(flags.keys())