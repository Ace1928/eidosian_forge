import collections
import os
import re
import sys
import functools
import itertools
def mac_ver(release='', versioninfo=('', '', ''), machine=''):
    """ Get macOS version information and return it as tuple (release,
        versioninfo, machine) with versioninfo being a tuple (version,
        dev_stage, non_release_version).

        Entries which cannot be determined are set to the parameter values
        which default to ''. All tuple entries are strings.
    """
    info = _mac_ver_xml()
    if info is not None:
        return info
    return (release, versioninfo, machine)