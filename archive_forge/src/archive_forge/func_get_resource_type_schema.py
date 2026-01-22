import json
import os
import sys
from oslo_utils import strutils
from glanceclient._i18n import _
from glanceclient.common import progressbar
from glanceclient.common import utils
from glanceclient import exc
from glanceclient.v2 import cache
from glanceclient.v2 import image_members
from glanceclient.v2 import image_schema
from glanceclient.v2 import images
from glanceclient.v2 import namespace_schema
from glanceclient.v2 import resource_type_schema
from glanceclient.v2 import tasks
def get_resource_type_schema():
    global RESOURCE_TYPE_SCHEMA
    if RESOURCE_TYPE_SCHEMA is None:
        schema_path = os.path.expanduser('~/.glanceclient/resource_type_schema.json')
        if os.path.isfile(schema_path):
            with open(schema_path, 'r') as f:
                schema_raw = f.read()
                RESOURCE_TYPE_SCHEMA = json.loads(schema_raw)
        else:
            return resource_type_schema.BASE_SCHEMA
    return RESOURCE_TYPE_SCHEMA