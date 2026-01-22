def patchOrUnpatchAll(*, patch, warn=True, raise_orjson=False):
    patchOrUnpatchJson(patch=patch, warn=warn)
    try:
        import orjson
    except ImportError:
        if raise_orjson:
            raise
    else:
        patchOrUnpatchOrjson(patch=patch, warn=warn)
    patchOrUnpatchMutableMappingSubclasshook(patch=patch, warn=warn)