from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _camel_to_snake
from ansible.module_utils._text import to_native
from datetime import datetime, timedelta
def get_auth_rules(self):
    result = dict()
    try:
        client = self._get_client()
        if self.type == 'namespace':
            rules = client.list_authorization_rules(self.resource_group, self.name)
        else:
            rules = client.list_authorization_rules(self.resource_group, self.namespace, self.name)
        while True:
            rule = rules.next()
            result[rule.name] = self.policy_to_dict(rule)
    except StopIteration:
        pass
    except Exception as exc:
        self.fail('Error when getting SAS policies for {0} {1}: {2}'.format(self.type, self.name, exc.message or str(exc)))
    return result