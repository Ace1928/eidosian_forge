from collections import defaultdict
import datetime
import io
import json
from prov import Error
from prov.serializers import Serializer
from prov.constants import *
from prov.model import (
import logging
def decode_json_document(content, document):
    bundles = dict()
    if 'bundle' in content:
        bundles = content['bundle']
        del content['bundle']
    decode_json_container(content, document)
    for bundle_id, bundle_content in bundles.items():
        bundle = ProvBundle(document=document)
        decode_json_container(bundle_content, bundle)
        document.add_bundle(bundle, bundle.valid_qualified_name(bundle_id))