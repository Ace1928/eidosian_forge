def patchOrUnpatchMutableMappingSubclasshook(*, patch, warn=True):
    warn_c = True
    if checkCExtension(warn=warn, warn_c=warn_c):
        return
    from importlib import import_module
    self = import_module(__name__)
    from collections.abc import MutableMapping
    from frozendict import frozendict
    if self._oldMutableMappingSubclasshook is None:
        if not patch:
            raise ValueError('Old MutableMapping subclasshook is None ' + '(maybe you already unpatched MutableMapping?)')
        oldMutableMappingSubclasshook = MutableMapping.__subclasshook__
    else:
        oldMutableMappingSubclasshook = self._oldMutableMappingSubclasshook
    if patch:

        @classmethod
        def frozendictMutableMappingSubclasshook(klass, subclass, *args, **kwargs):
            if klass == MutableMapping:
                if issubclass(subclass, frozendict):
                    return False
                return oldMutableMappingSubclasshook(subclass, *args, **kwargs)
            return NotImplemented
        defaultMutableMappingSubclasshook = frozendictMutableMappingSubclasshook
        newOldMutableMappingSubclasshook = oldMutableMappingSubclasshook
    else:
        defaultMutableMappingSubclasshook = oldMutableMappingSubclasshook
        newOldMutableMappingSubclasshook = None
    self._oldMutableMappingSubclasshook = newOldMutableMappingSubclasshook
    MutableMapping.__subclasshook__ = defaultMutableMappingSubclasshook
    try:
        MutableMapping._abc_caches_clear()
    except AttributeError:
        MutableMapping._abc_cache.discard(frozendict)
        MutableMapping._abc_negative_cache.discard(frozendict)