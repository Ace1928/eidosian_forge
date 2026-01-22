import sys
import os
import builtins
import _sitebuiltins
import io
import stat
def check_enableusersite():
    """Check if user site directory is safe for inclusion

    The function tests for the command line flag (including environment var),
    process uid/gid equal to effective uid/gid.

    None: Disabled for security reasons
    False: Disabled by user (command line option)
    True: Safe and enabled
    """
    if sys.flags.no_user_site:
        return False
    if hasattr(os, 'getuid') and hasattr(os, 'geteuid'):
        if os.geteuid() != os.getuid():
            return None
    if hasattr(os, 'getgid') and hasattr(os, 'getegid'):
        if os.getegid() != os.getgid():
            return None
    return True