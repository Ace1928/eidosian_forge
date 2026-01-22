from collections import OrderedDict
from copy import deepcopy
from pickle import dumps
import os
import getpass
import platform
from uuid import uuid1
import simplejson as json
import numpy as np
import prov.model as pm
from .. import get_info, logging, __version__
from .filemanip import md5, hashlib, hash_infile
def prov_encode(graph, value, create_container=True):
    if isinstance(value, (list, tuple)) and create_container:
        value = list(value)
        if len(value) == 0:
            encoded_literal = safe_encode(value)
            attr = {pm.PROV['value']: encoded_literal}
            eid = get_attr_id(attr)
            return graph.entity(eid, attr)
        if len(value) == 1:
            return prov_encode(graph, value[0])
        entities = []
        for item in value:
            item_entity = prov_encode(graph, item)
            entities.append(item_entity)
            if isinstance(item, (list, tuple)):
                continue
            item_entity_val = list(item_entity.value)[0]
            is_str = isinstance(item_entity_val, str)
            if not is_str or (is_str and 'file://' not in item_entity_val):
                return prov_encode(graph, value, create_container=False)
        eid = get_id()
        entity = graph.collection(identifier=eid)
        for item_entity in entities:
            graph.hadMember(eid, item_entity)
        return entity
    else:
        encoded_literal = safe_encode(value)
        attr = {pm.PROV['value']: encoded_literal}
        if isinstance(value, str) and os.path.exists(value):
            attr.update({pm.PROV['location']: encoded_literal})
            if not os.path.isdir(value):
                sha512 = hash_infile(value, crypto=hashlib.sha512)
                attr.update({crypto['sha512']: pm.Literal(sha512, pm.XSD['string'])})
                eid = get_attr_id(attr, skip=[pm.PROV['location'], pm.PROV['value']])
            else:
                eid = get_attr_id(attr, skip=[pm.PROV['location']])
        else:
            eid = get_attr_id(attr)
        entity = graph.entity(eid, attr)
    return entity