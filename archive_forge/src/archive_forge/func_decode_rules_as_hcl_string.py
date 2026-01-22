from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import to_text, AnsibleModule
def decode_rules_as_hcl_string(rules_as_hcl):
    """
    Converts the given HCL (string) representation of rules into a list of rule domain models.
    :param rules_as_hcl: the HCL (string) representation of a collection of rules
    :return: the equivalent domain model to the given rules
    """
    rules_as_hcl = to_text(rules_as_hcl)
    rules_as_json = hcl.loads(rules_as_hcl)
    return decode_rules_as_json(rules_as_json)