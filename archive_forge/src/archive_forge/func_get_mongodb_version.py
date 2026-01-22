import os
import testinfra.utils.ansible_runner
def get_mongodb_version(host):
    return include_vars(host)['ansible_facts']['mongodb_version']