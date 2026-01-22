from __future__ import absolute_import, division, print_function
import traceback
import copy
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.kubernetes.core.plugins.module_utils.helm import (
from ansible_collections.kubernetes.core.plugins.module_utils.helm_args_common import (
def install_repository(command, repository_name, repository_url, repository_username, repository_password, pass_credentials, force_update):
    install_command = command + ' repo add ' + repository_name + ' ' + repository_url
    if repository_username is not None and repository_password is not None:
        install_command += ' --username=' + repository_username
        install_command += ' --password=' + repository_password
    if pass_credentials:
        install_command += ' --pass-credentials'
    if force_update:
        install_command += ' --force-update'
    return install_command