import dill
from enum import EnumMeta
import sys
from collections import namedtuple
def test_class_objects():
    clslist = [_class, _class2, _newclass, _newclass2, _mclass]
    objlist = [o, oc, n, nc, m]
    _clslist = [dill.dumps(obj) for obj in clslist]
    _objlist = [dill.dumps(obj) for obj in objlist]
    for obj in clslist:
        globals().pop(obj.__name__)
    del clslist
    for obj in ['o', 'oc', 'n', 'nc']:
        globals().pop(obj)
    del objlist
    del obj
    for obj, cls in zip(_objlist, _clslist):
        _cls = dill.loads(cls)
        _obj = dill.loads(obj)
        assert _obj.ok()
        assert _cls.ok(_cls())
        if _cls.__name__ == '_mclass':
            assert type(_cls).__name__ == '_meta'