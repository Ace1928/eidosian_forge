from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def rm_repo(module, repo):
    """remove a repository"""
    apt_repo(module, 'rm', repo)