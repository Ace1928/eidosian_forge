import json
import numpy as np
from psycopg2 import connect
from psycopg2.extras import execute_values
from ase.db.sqlite import (init_statements, index_statements, VERSION,
from ase.io.jsonio import (encode as ase_encode,
def insert_ase_and_ndarray_objects(obj):
    if isinstance(obj, dict):
        objtype = obj.pop('__ase_objtype__', None)
        if objtype is not None:
            return create_ase_object(objtype, insert_ase_and_ndarray_objects(obj))
        data = obj.get('__ndarray__')
        if data is not None:
            return create_ndarray(*data)
        return {key: insert_ase_and_ndarray_objects(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [insert_ase_and_ndarray_objects(value) for value in obj]
    return obj