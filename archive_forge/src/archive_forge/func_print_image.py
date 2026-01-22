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
def print_image(image_obj, human_readable=False, max_col_width=None):
    ignore = ['self', 'access', 'file', 'schema']
    image = dict([item for item in image_obj.items() if item[0] not in ignore])
    if 'virtual_size' in image:
        image['virtual_size'] = image.get('virtual_size') or 'Not available'
    if human_readable:
        image['size'] = make_size_human_readable(image['size'])
    if str(max_col_width).isdigit():
        print_dict(image, max_column_width=max_col_width)
    else:
        print_dict(image)