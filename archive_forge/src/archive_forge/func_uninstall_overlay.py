from __future__ import absolute_import, division, print_function
import shutil
import traceback
from os import path
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
def uninstall_overlay(module, name):
    """Uninstalls the given overlay repository from the system.

    :param name: the overlay id to uninstall

    :returns: True if the overlay was uninstalled, or False if doesn't exist
        (i.e. nothing has changed)
    :raises ModuleError
    """
    layman = init_layman()
    if not layman.is_installed(name):
        return False
    if module.check_mode:
        mymsg = "Would remove layman repo '" + name + "'"
        module.exit_json(changed=True, msg=mymsg)
    layman.delete_repos(name)
    if layman.get_errors():
        raise ModuleError(layman.get_errors())
    return True