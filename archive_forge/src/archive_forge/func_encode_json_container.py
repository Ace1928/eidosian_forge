from collections import defaultdict
import datetime
import io
import json
from prov import Error
from prov.serializers import Serializer
from prov.constants import *
from prov.model import (
import logging
def encode_json_container(bundle):
    container = defaultdict(dict)
    prefixes = {}
    for namespace in bundle._namespaces.get_registered_namespaces():
        prefixes[namespace.prefix] = namespace.uri
    if bundle._namespaces._default:
        prefixes['default'] = bundle._namespaces._default.uri
    if prefixes:
        container['prefix'] = prefixes
    id_generator = AnonymousIDGenerator()

    def real_or_anon_id(r):
        return r._identifier if r._identifier else id_generator.get_anon_id(r)
    for record in bundle._records:
        rec_type = record.get_type()
        rec_label = PROV_N_MAP[rec_type]
        identifier = str(real_or_anon_id(record))
        record_json = {}
        if record._attributes:
            for attr, values in record._attributes.items():
                if not values:
                    continue
                attr_name = str(attr)
                if attr in PROV_ATTRIBUTE_QNAMES:
                    record_json[attr_name] = str(first(values))
                elif attr in PROV_ATTRIBUTE_LITERALS:
                    record_json[attr_name] = first(values).isoformat()
                elif len(values) == 1:
                    record_json[attr_name] = encode_json_representation(first(values))
                else:
                    record_json[attr_name] = list((encode_json_representation(value) for value in values))
        if identifier not in container[rec_label]:
            container[rec_label][identifier] = record_json
        else:
            current_content = container[rec_label][identifier]
            if hasattr(current_content, 'items'):
                container[rec_label][identifier] = [current_content]
            container[rec_label][identifier].append(record_json)
    return container