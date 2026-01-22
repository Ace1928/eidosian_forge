from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def clc_install_package(self, server, package_id, package_params):
    """
        Install the package to a given clc server
        :param server: The server object where the package needs to be installed
        :param package_id: The blue print package id
        :param package_params: the required argument dict for the package installation
        :return: The result object from the CLC API call
        """
    result = None
    try:
        result = server.ExecutePackage(package_id=package_id, parameters=package_params)
    except CLCException as ex:
        self.module.fail_json(msg='Failed to install package : {0} to server {1}. {2}'.format(package_id, server.id, ex.message))
    return result