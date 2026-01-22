import csv
import gzip
import json
from nltk.internals import deprecated
def get_header_field_list(main_fields, entity_type, entity_fields):
    if _is_composed_key(entity_type):
        key, value = _get_key_value_composed(entity_type)
        main_entity = key
        sub_entity = value
    else:
        main_entity = None
        sub_entity = entity_type
    if main_entity:
        output1 = [HIER_SEPARATOR.join([main_entity, x]) for x in main_fields]
    else:
        output1 = main_fields
    output2 = [HIER_SEPARATOR.join([sub_entity, x]) for x in entity_fields]
    return output1 + output2