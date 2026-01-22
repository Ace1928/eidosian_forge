from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from distutils.version import LooseVersion
def is_invalid_gluster_version(module, required_version):
    cmd = module.get_bin_path('gluster', True) + ' --version'
    result = module.run_command(cmd)
    ver_line = result[1].split('\n')[0]
    version = ver_line.split(' ')[1]
    return LooseVersion(version) < LooseVersion(required_version)