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
def convert_list_props_to_comma_separated(data, props=None):
    """Convert the list-type properties to comma-separated strings

    :param data: the input dict object.
    :param props: the properties whose values will be converted.
        Default to None to convert all list-type properties of the input.
    :returns: the result dict instance.
    """
    result = dict(data)
    if props is None:
        props = data.keys()
    for prop in props:
        val = data.get(prop, None)
        if isinstance(val, list):
            result[prop] = ', '.join(map(str, val))
    return result