from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import to_text, AnsibleModule
def decode_rules_as_json(rules_as_json):
    """
    Converts the given JSON representation of rules into a list of rule domain models.
    :param rules_as_json: the JSON representation of a collection of rules
    :return: the equivalent domain model to the given rules
    """
    rules = RuleCollection()
    for scope in rules_as_json:
        if not isinstance(rules_as_json[scope], dict):
            rules.add(Rule(scope, rules_as_json[scope]))
        else:
            for pattern, policy in rules_as_json[scope].items():
                rules.add(Rule(scope, policy[_POLICY_JSON_PROPERTY], pattern))
    return rules