from __future__ import (absolute_import, division, print_function)
import re
def create_containers_and_parents(container_dn):
    """Create a container and if needed the parents containers"""
    import univention.admin.uexceptions as uexcp
    if not container_dn.startswith('cn='):
        raise AssertionError()
    try:
        parent = ldap_dn_tree_parent(container_dn)
        obj = umc_module_for_add('container/cn', parent)
        obj['name'] = container_dn.split(',')[0].split('=')[1]
        obj['description'] = 'container created by import'
    except uexcp.ldapError:
        create_containers_and_parents(parent)
        obj = umc_module_for_add('container/cn', parent)
        obj['name'] = container_dn.split(',')[0].split('=')[1]
        obj['description'] = 'container created by import'