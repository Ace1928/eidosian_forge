from __future__ import annotations
from .. import mesonlib
from .. import mlog
from .common import cmake_is_debug
import typing as T
def target_file(arg: str) -> str:
    if arg not in trace.targets:
        mlog.warning(f"Unable to evaluate the cmake variable '$<TARGET_FILE:{arg}>'.")
        return ''
    tgt = trace.targets[arg]
    cfgs = []
    cfg = ''
    if 'IMPORTED_CONFIGURATIONS' in tgt.properties:
        cfgs = [x for x in tgt.properties['IMPORTED_CONFIGURATIONS'] if x]
        cfg = cfgs[0]
    if cmake_is_debug(trace.env):
        if 'DEBUG' in cfgs:
            cfg = 'DEBUG'
        elif 'RELEASE' in cfgs:
            cfg = 'RELEASE'
    elif 'RELEASE' in cfgs:
        cfg = 'RELEASE'
    if f'IMPORTED_IMPLIB_{cfg}' in tgt.properties:
        return ';'.join([x for x in tgt.properties[f'IMPORTED_IMPLIB_{cfg}'] if x])
    elif 'IMPORTED_IMPLIB' in tgt.properties:
        return ';'.join([x for x in tgt.properties['IMPORTED_IMPLIB'] if x])
    elif f'IMPORTED_LOCATION_{cfg}' in tgt.properties:
        return ';'.join([x for x in tgt.properties[f'IMPORTED_LOCATION_{cfg}'] if x])
    elif 'IMPORTED_LOCATION' in tgt.properties:
        return ';'.join([x for x in tgt.properties['IMPORTED_LOCATION'] if x])
    return ''