import os
from ... import LooseVersion
from ...utils.filemanip import fname_presuffix
from ..base import (
@classmethod
def looseversion(cls):
    """Return a comparable version object

        If no version found, use LooseVersion('0.0.0')
        """
    ver = cls.version()
    if ver is None:
        return LooseVersion('0.0.0')
    vinfo = ver.rstrip().split('-')
    try:
        int(vinfo[-1], 16)
    except ValueError:
        githash = ''
    else:
        githash = '.' + vinfo[-1]
    if githash:
        if vinfo[3] == 'dev':
            vstr = '6.0.0-dev' + githash
        elif vinfo[5][0] == 'v':
            vstr = vinfo[5][1:]
        elif len([1 for val in vinfo[3] if val == '.']) == 2:
            'version string: freesurfer-linux-centos7_x86_64-7.1.0-20200511-813297b'
            vstr = vinfo[3]
        else:
            raise RuntimeError('Unknown version string: ' + ver)
    elif 'dev' in ver:
        vstr = vinfo[-1] + '-dev'
    else:
        vstr = ver.rstrip().split('-v')[-1]
    return LooseVersion(vstr)