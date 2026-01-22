import datetime
import errno
import os
import os.path
import time
def get_user_config_directory():
    """Returns a platform-specific root directory for user config settings."""
    if os.name == 'nt':
        appdata = os.getenv('LOCALAPPDATA')
        if appdata:
            return appdata
        appdata = os.getenv('APPDATA')
        if appdata:
            return appdata
        return None
    xdg_config_home = os.getenv('XDG_CONFIG_HOME')
    if xdg_config_home:
        return xdg_config_home
    return os.path.join(os.path.expanduser('~'), '.config')