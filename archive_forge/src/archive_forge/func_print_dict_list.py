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
def print_dict_list(objects, fields):
    pt = prettytable.PrettyTable([f for f in fields], caching=False)
    pt.align = 'l'
    for o in objects:
        row = []
        for field in fields:
            field_name = field.lower().replace(' ', '_')
            if field_name == 'task_id':
                field_name = 'id'
            data = o.get(field_name, '')
            row.append(data)
        pt.add_row(row)
    print(encodeutils.safe_decode(pt.get_string()))