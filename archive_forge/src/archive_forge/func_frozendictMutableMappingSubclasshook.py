@classmethod
def frozendictMutableMappingSubclasshook(klass, subclass, *args, **kwargs):
    if klass == MutableMapping:
        if issubclass(subclass, frozendict):
            return False
        return oldMutableMappingSubclasshook(subclass, *args, **kwargs)
    return NotImplemented