import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def build_option_string(options):
    """Format a string of option flags (--key 'value').

    This will quote the values, in case spaces are included.
    Any values that are None are excluded entirely.

    Usage::

        build_option_string({
            "--email": "me@example.com",
            "--name": "example.com."
            "--ttl": None,

        })

    Returns::

        "--email 'me@example.com' --name 'example.com.'
    """
    return ' '.join((f"{flag} '{value}'" for flag, value in options.items() if value is not None))