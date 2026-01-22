import argparse
import base64
import contextlib
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from oslo_utils import strutils
import yaml
from ironicclient.common.i18n import _
from ironicclient import exc
def key_value_pairs_to_dict(key_value_pairs):
    """Convert a list of key-value pairs to a dictionary.

    :param key_value_pairs: a list of strings, each string is in the form
                            <key>=<value>
    :returns: a dictionary, possibly empty
    """
    if key_value_pairs:
        return dict((split_and_deserialize(v) for v in key_value_pairs))
    return {}