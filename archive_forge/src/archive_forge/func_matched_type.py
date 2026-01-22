from __future__ import print_function, absolute_import
from shibokensupport.signature import inspect
from shibokensupport.signature import get_signature
from shibokensupport.signature.mapping import update_mapping, namespace
from textwrap import dedent
def matched_type(args, sigs):
    for sig in sigs:
        params = list(sig.parameters.values())
        if len(args) > len(params):
            continue
        if len(args) < len(params):
            k = len(args)
            if params[k].default is params[k].empty:
                continue
        ok = True
        for arg, param in zip(args, params):
            ann = param.annotation
            if qt_isinstance(arg, ann):
                continue
            ok = False
        if ok:
            return sig
    return None