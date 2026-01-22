import string
import textwrap
from ansible.module_utils._text import to_native
from ansible.module_utils.six.moves.urllib import parse as urlparse
def tower_callback_script(tower_address, job_template_id, host_config_key, windows, passwd):
    if windows:
        return to_native(_windows_callback_script(passwd=passwd))
    return _linux_callback_script(tower_address, job_template_id, host_config_key)