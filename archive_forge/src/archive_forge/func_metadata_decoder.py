from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
def metadata_decoder(metadata):
    items = {}
    if 'items' in metadata:
        metadata_items = metadata['items']
        for item in metadata_items:
            items[item['key']] = item['value']
    return items