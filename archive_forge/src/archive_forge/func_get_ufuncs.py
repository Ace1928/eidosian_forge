import numpy as np
def get_ufuncs():
    """obtain a list of supported ufuncs in the db"""
    _lazy_init_db()
    return _ufunc_db.keys()