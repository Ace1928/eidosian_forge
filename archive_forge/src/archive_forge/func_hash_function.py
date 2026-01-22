from abc import ABC
import inspect
import hashlib
@staticmethod
def hash_function(fn, callback_id=''):
    fn_source = inspect.getsource(fn)
    fn_str = fn_source
    return hashlib.sha1(callback_id.encode('utf-8') + fn_str.encode('utf-8')).hexdigest()