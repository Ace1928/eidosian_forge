from __future__ import absolute_import, division, print_function
import shutil
import traceback
from os import path
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
def install_overlay(module, name, list_url=None):
    """Installs the overlay repository. If not on the central overlays list,
    then :list_url of an alternative list must be provided. The list will be
    fetched and saved under ``%(overlay_defs)/%(name.xml)`` (location of the
    ``overlay_defs`` is read from the Layman's configuration).

    :param name: the overlay id
    :param list_url: the URL of the remote repositories list to look for the overlay
        definition (optional, default: None)

    :returns: True if the overlay was installed, or False if already exists
        (i.e. nothing has changed)
    :raises ModuleError
    """
    layman_conf = BareConfig(read_configfile=True)
    layman = init_layman(layman_conf)
    if layman.is_installed(name):
        return False
    if module.check_mode:
        mymsg = "Would add layman repo '" + name + "'"
        module.exit_json(changed=True, msg=mymsg)
    if not layman.is_repo(name):
        if not list_url:
            raise ModuleError("Overlay '%s' is not on the list of known overlays and URL of the remote list was not provided." % name)
        overlay_defs = layman_conf.get_option('overlay_defs')
        dest = path.join(overlay_defs, name + '.xml')
        download_url(module, list_url, dest)
        layman = init_layman()
    if not layman.add_repos(name):
        raise ModuleError(layman.get_errors())
    return True