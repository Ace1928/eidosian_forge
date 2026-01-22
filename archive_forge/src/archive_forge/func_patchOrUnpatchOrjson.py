def patchOrUnpatchOrjson(*, patch, warn=True):
    if not checkCExtension(warn=warn):
        return
    from importlib import import_module
    self = import_module(__name__)
    import orjson
    if self._oldOrjsonDumps is None:
        if not patch:
            raise ValueError('Old orjson encoder is None ' + '(maybe you already unpatched orjson?)')
        oldOrjsonDumps = orjson.dumps
    else:
        oldOrjsonDumps = self._oldOrjsonDumps
    if patch:
        from frozendict import frozendict

        def frozendictOrjsonDumps(obj, *args, **kwargs):
            if isinstance(obj, frozendict):
                obj = dict(obj)
            return oldOrjsonDumps(obj, *args, **kwargs)
        defaultOrjsonDumps = frozendictOrjsonDumps
        newOldOrjsonDumps = oldOrjsonDumps
    else:
        defaultOrjsonDumps = oldOrjsonDumps
        newOldOrjsonDumps = None
    self._oldOrjsonDumps = newOldOrjsonDumps
    orjson.dumps = defaultOrjsonDumps
    orjson.orjson.dumps = defaultOrjsonDumps