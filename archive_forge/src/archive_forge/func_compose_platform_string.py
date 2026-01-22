from __future__ import (absolute_import, division, print_function)
import re
def compose_platform_string(os=None, arch=None, variant=None, daemon_os=None, daemon_arch=None):
    if os is None and daemon_os is not None:
        os = _normalize_os(daemon_os)
    if arch is None and daemon_arch is not None:
        arch, variant = _normalize_arch(daemon_arch, variant or '')
        variant = variant or None
    return str(_Platform(os=os, arch=arch, variant=variant or None))