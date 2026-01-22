from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_enum_kdctype(kerberos_domain_controller_type):
    """Getting correct enum values for kerberos_domain_controller_type
        :param: kerberos_domain_controller_type: Type of Kerberos Domain Controller used for secure NFS service.
        :return: enum value for kerberos_domain_controller_type.
    """
    if utils.KdcTypeEnum[kerberos_domain_controller_type]:
        kerberos_domain_controller_type = utils.KdcTypeEnum[kerberos_domain_controller_type]
        return kerberos_domain_controller_type