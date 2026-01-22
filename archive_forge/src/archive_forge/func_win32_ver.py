import collections
import os
import re
import sys
import functools
import itertools
def win32_ver(release='', version='', csd='', ptype=''):
    try:
        from sys import getwindowsversion
    except ImportError:
        return (release, version, csd, ptype)
    winver = getwindowsversion()
    try:
        major, minor, build = map(int, _syscmd_ver()[2].split('.'))
    except ValueError:
        major, minor, build = winver.platform_version or winver[:3]
    version = '{0}.{1}.{2}'.format(major, minor, build)
    release = _WIN32_CLIENT_RELEASES.get((major, minor)) or _WIN32_CLIENT_RELEASES.get((major, None)) or release
    if winver[:2] == (major, minor):
        try:
            csd = 'SP{}'.format(winver.service_pack_major)
        except AttributeError:
            if csd[:13] == 'Service Pack ':
                csd = 'SP' + csd[13:]
    if getattr(winver, 'product_type', None) == 3:
        release = _WIN32_SERVER_RELEASES.get((major, minor)) or _WIN32_SERVER_RELEASES.get((major, None)) or release
    try:
        try:
            import winreg
        except ImportError:
            import _winreg as winreg
    except ImportError:
        pass
    else:
        try:
            cvkey = 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion'
            with winreg.OpenKeyEx(winreg.HKEY_LOCAL_MACHINE, cvkey) as key:
                ptype = winreg.QueryValueEx(key, 'CurrentType')[0]
        except OSError:
            pass
    return (release, version, csd, ptype)