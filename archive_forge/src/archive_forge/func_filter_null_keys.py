import logging
import uuid
def filter_null_keys(dictionary):
    return dict(((k, v) for k, v in dictionary.items() if v is not None))