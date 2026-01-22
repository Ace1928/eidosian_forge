from __future__ import (absolute_import, division, print_function)
import re
@classmethod
def parse_platform_string(cls, string, daemon_os=None, daemon_arch=None):
    if string is None:
        return cls()
    if not string:
        raise ValueError('Platform string must be non-empty')
    parts = string.split('/', 2)
    arch = None
    variant = None
    if len(parts) == 1:
        _validate_part(string, string, 'OS/architecture')
        os = _normalize_os(string)
        if os in _KNOWN_OS:
            if daemon_arch is not None:
                arch, variant = _normalize_arch(daemon_arch, '')
            return cls(os=os, arch=arch, variant=variant)
        arch, variant = _normalize_arch(os, '')
        if arch in _KNOWN_ARCH:
            return cls(os=_normalize_os(daemon_os) if daemon_os else None, arch=arch or None, variant=variant or None)
        raise ValueError('Invalid platform string "{0}": unknown OS or architecture'.format(string))
    os = _validate_part(string, parts[0], 'OS')
    if not os:
        raise ValueError('Invalid platform string "{0}": OS is empty'.format(string))
    arch = _validate_part(string, parts[1], 'architecture') if len(parts) > 1 else None
    if arch is not None and (not arch):
        raise ValueError('Invalid platform string "{0}": architecture is empty'.format(string))
    variant = _validate_part(string, parts[2], 'variant') if len(parts) > 2 else None
    if variant is not None and (not variant):
        raise ValueError('Invalid platform string "{0}": variant is empty'.format(string))
    arch, variant = _normalize_arch(arch, variant or '')
    if len(parts) == 2 and arch == 'arm' and (variant == 'v7'):
        variant = None
    if len(parts) == 3 and arch == 'arm64' and (variant == ''):
        variant = 'v8'
    return cls(os=_normalize_os(os), arch=arch, variant=variant or None)