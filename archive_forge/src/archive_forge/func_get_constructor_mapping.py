import json
import os
import threading
def get_constructor_mapping():
    global _class_mapping
    if _class_mapping is not None:
        return _class_mapping.copy()
    with _class_mapping_lock:
        if _class_mapping is not None:
            return _class_mapping.copy()
        tmp_class_mapping = {}
        with open(_json_path, 'r') as json_file:
            tmp_class_mapping.update(json.load(json_file))
        _class_mapping = tmp_class_mapping
        return tmp_class_mapping.copy()