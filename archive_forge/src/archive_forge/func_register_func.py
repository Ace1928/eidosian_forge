from abc import ABC
import inspect
import hashlib
@staticmethod
def register_func(fn, progress, callback_id):
    key = BaseLongCallbackManager.hash_function(fn, callback_id)
    BaseLongCallbackManager.functions.append((key, fn, progress))
    for manager in BaseLongCallbackManager.managers:
        manager.register(key, fn, progress)
    return key