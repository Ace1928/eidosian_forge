import json
import numpy as np
from psycopg2 import connect
from psycopg2.extras import execute_values
from ase.db.sqlite import (init_statements, index_statements, VERSION,
from ase.io.jsonio import (encode as ase_encode,
def insert_nan_and_inf(obj):
    if isinstance(obj, dict) and '__special_number__' in obj:
        return float(obj['__special_number__'])
    if isinstance(obj, list):
        return [insert_nan_and_inf(x) for x in obj]
    if isinstance(obj, dict):
        return {key: insert_nan_and_inf(value) for key, value in obj.items()}
    return obj