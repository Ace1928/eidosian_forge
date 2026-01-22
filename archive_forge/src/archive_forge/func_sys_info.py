import os
import platform
import pprint
import sys
import subprocess
from pathlib import Path
from IPython.core import release
from IPython.utils import _sysinfo, encoding
def sys_info():
    """Return useful information about IPython and the system, as a string.

    Examples
    --------
    ::
    
        In [2]: print(sys_info())
        {'commit_hash': '144fdae',      # random
         'commit_source': 'repository',
         'ipython_path': '/home/fperez/usr/lib/python2.6/site-packages/IPython',
         'ipython_version': '0.11.dev',
         'os_name': 'posix',
         'platform': 'Linux-2.6.35-22-generic-i686-with-Ubuntu-10.10-maverick',
         'sys_executable': '/usr/bin/python',
         'sys_platform': 'linux2',
         'sys_version': '2.6.6 (r266:84292, Sep 15 2010, 15:52:39) \\n[GCC 4.4.5]'}
    """
    return pprint.pformat(get_sys_info())