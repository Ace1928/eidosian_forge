import os
from os.path import sep
def query_xinput():
    global _cache_xinput
    if _cache_xinput is None:
        _cache_xinput = []
        devids = getout('xinput', '--list', '--id-only')
        for did in devids.splitlines():
            devprops = getout('xinput', '--list-props', did)
            evpath = None
            for prop in devprops.splitlines():
                prop = prop.strip()
                if prop.startswith(b'Device Enabled') and prop.endswith(b'0'):
                    evpath = None
                    break
                if prop.startswith(b'Device Node'):
                    try:
                        evpath = prop.split('"')[1]
                    except Exception:
                        evpath = None
            if evpath:
                _cache_xinput.append(evpath)