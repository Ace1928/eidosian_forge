import base64
import inspect
import builtins
@classmethod
def cls_from_jsondict_key(cls, k):
    import sys
    mod = sys.modules[cls.__module__]
    return getattr(mod, k)