import datetime
import errno
import functools
import hashlib
import json
import os
import re
import sys
import threading
import urllib.parse
import uuid
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
import wrapt
from glanceclient._i18n import _
from glanceclient import exc
def on_data_require_fields(data_fields, required=REQUIRED_FIELDS_ON_DATA):
    """Decorator to check commands' validity

    This decorator checks that required fields are present when image
    data has been supplied via command line arguments or via stdin

    On error throws CommandError exception with meaningful message.

    :param data_fields: Which fields' presence imply image data
    :type data_fields: iter
    :param required: Required fields
    :type required: iter
    :return: function decorator
    """

    def args_decorator(func):

        def prepare_fields(fields):
            args = ('--' + x.replace('_', '-') for x in fields)
            return ', '.join(args)

        @functools.wraps(func)
        def func_wrapper(gc, args):
            fields = set((a[0] for a in vars(args).items() if a[1]))
            present = fields.intersection(data_fields)
            missing = set(required) - fields
            if (present or get_data_file(args)) and missing:
                msg = _('error: Must provide %(req)s when using %(opt)s.') % {'req': prepare_fields(missing), 'opt': prepare_fields(present) or 'stdin'}
                raise exc.CommandError(msg)
            return func(gc, args)
        return func_wrapper
    return args_decorator