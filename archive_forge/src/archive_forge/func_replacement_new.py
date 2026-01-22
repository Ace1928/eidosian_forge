import collections
import inspect
from neutron_lib._i18n import _
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
@staticmethod
def replacement_new(cls, *args, **kwargs):
    if new_inherited:
        super_new = super(klass, cls).__new__
        if super_new is object.__new__:
            instance = super_new(cls)
        else:
            instance = super_new(cls, *args, **kwargs)
    else:
        instance = orig_new(cls, *args, **kwargs)
    if getattr(instance, '_DECORATED_METHODS_SUBSCRIBED', False):
        return instance
    for name, unbound_method in inspect.getmembers(cls):
        if not inspect.ismethod(unbound_method) and (not inspect.isfunction(unbound_method)):
            continue
        func = getattr(unbound_method, 'im_func', unbound_method)
        if func not in _REGISTERED_CLASS_METHODS:
            continue
        for resource, event, priority in _REGISTERED_CLASS_METHODS[func]:
            subscribe(getattr(instance, name), resource, event, priority)
    setattr(instance, '_DECORATED_METHODS_SUBSCRIBED', True)
    return instance