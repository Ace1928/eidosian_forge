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
def handle_json_or_file_arg(json_arg):
    """Attempts to read JSON argument from file or string.

    :param json_arg: May be a file name containing the YAML or JSON, or
        a JSON string.
    :returns: A list or dictionary parsed from JSON.
    :raises: InvalidAttribute if the argument cannot be parsed.
    """
    if os.path.isfile(json_arg):
        try:
            with open(json_arg, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            err = _("Cannot get JSON/YAML from file '%(file)s'. Error: %(err)s") % {'err': e, 'file': json_arg}
            raise exc.InvalidAttribute(err)
    try:
        json_arg = json.loads(json_arg)
    except ValueError as e:
        err = _("Value '%(string)s' is not a file and cannot be parsed as JSON: '%(err)s'") % {'err': e, 'string': json_arg}
        raise exc.InvalidAttribute(err)
    return json_arg