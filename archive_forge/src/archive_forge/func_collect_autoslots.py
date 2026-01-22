import types
from collections import namedtuple
from copy import deepcopy
from weakref import ref as _weakref_ref
@staticmethod
def collect_autoslots(cls):
    has_dict = '__dict__' in dir(cls.__mro__[0])
    slots = []
    seen = set()
    for c in reversed(cls.__mro__):
        for slot in getattr(c, '__slots__', ()):
            if slot in seen:
                continue
            if slot in AutoSlots._ignore_slots:
                continue
            seen.add(slot)
            slots.append(slot)
    slots = tuple(slots)
    slot_mappers = {}
    dict_mappers = {}
    for c in reversed(cls.__mro__):
        for slot, mapper in getattr(c, '__autoslot_mappers__', {}).items():
            if slot in seen:
                slot_mappers[slots.index(slot)] = mapper
            else:
                dict_mappers[slot] = mapper
    cls.__auto_slots__ = _autoslot_info(has_dict, slots, slot_mappers, dict_mappers)