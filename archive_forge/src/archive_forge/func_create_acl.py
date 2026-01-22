from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import to_text, AnsibleModule
def create_acl(consul_client, configuration):
    """
    Creates an ACL.
    :param consul_client: the consul client
    :param configuration: the run configuration
    :return: the output of the creation
    """
    rules_as_hcl = encode_rules_as_hcl_string(configuration.rules) if len(configuration.rules) > 0 else None
    token = consul_client.acl.create(name=configuration.name, type=configuration.token_type, rules=rules_as_hcl, acl_id=configuration.token)
    rules = configuration.rules
    return Output(changed=True, token=token, rules=rules, operation=CREATE_OPERATION)