def patchOrUnpatchJson(*, patch, warn=True):
    if not checkCExtension(warn=warn):
        return
    from importlib import import_module
    self = import_module(__name__)
    import frozendict as cool
    import json
    OldJsonEncoder = self._OldJsonEncoder
    FrozendictJsonEncoder = cool._getFrozendictJsonEncoder(OldJsonEncoder)
    if patch:
        DefaultJsonEncoder = FrozendictJsonEncoder
    else:
        DefaultJsonEncoder = OldJsonEncoder
    if DefaultJsonEncoder is None:
        default_json_encoder = None
    else:
        default_json_encoder = DefaultJsonEncoder(skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, indent=None, separators=None, default=None)
    if patch:
        if OldJsonEncoder is None:
            self._OldJsonEncoder = json.encoder.JSONEncoder
    else:
        if OldJsonEncoder is None:
            raise ValueError('Old json encoder is None ' + '(maybe you already unpatched json?)')
        self._OldJsonEncoder = None
    cool.FrozendictJsonEncoder = FrozendictJsonEncoder
    json.JSONEncoder = DefaultJsonEncoder
    json.encoder.JSONEncoder = DefaultJsonEncoder
    json._default_encoder = default_json_encoder