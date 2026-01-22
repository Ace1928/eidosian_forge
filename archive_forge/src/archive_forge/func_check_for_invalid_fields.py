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
def check_for_invalid_fields(fields, valid_fields):
    """Check for invalid fields.

    :param fields: A list of fields specified by the user.
    :param valid_fields: A list of valid fields.
    :raises CommandError: If invalid fields were specified by the user.
    """
    if not fields:
        return
    invalid_fields = set(fields) - set(valid_fields)
    if invalid_fields:
        raise exc.CommandError(_('Invalid field(s) requested: %(invalid)s. Valid fields are: %(valid)s.') % {'invalid': ', '.join(invalid_fields), 'valid': ', '.join(valid_fields)})