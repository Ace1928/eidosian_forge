from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.crypto.plugins.module_utils.acme.acme import (
from ansible_collections.community.crypto.plugins.module_utils.acme.account import (
from ansible_collections.community.crypto.plugins.module_utils.acme.challenges import (
from ansible_collections.community.crypto.plugins.module_utils.acme.certificates import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.io import (
from ansible_collections.community.crypto.plugins.module_utils.acme.orders import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
def finish_challenges(self):
    """
        Verify challenges for all identifiers of the CSR.
        """
    self.authorizations = {}
    if self.version == 1:
        for identifier_type, identifier in self.identifiers:
            authz = Authorization.create(self.client, identifier_type, identifier)
            self.authorizations[combine_identifier(identifier_type, identifier)] = authz
    else:
        self.order = Order.from_url(self.client, self.order_uri)
        self.order.load_authorizations(self.client)
        self.authorizations.update(self.order.authorizations)
    authzs_to_wait_for = []
    for type_identifier, authz in self.authorizations.items():
        if authz.status == 'pending':
            if self.challenge is not None:
                authz.call_validate(self.client, self.challenge, wait=False)
                authzs_to_wait_for.append(authz)
            elif authz.status != 'valid':
                authz.raise_error('Status is not "valid", even though no challenge should be necessary', module=self.client.module)
            self.changed = True
    wait_for_validation(authzs_to_wait_for, self.client)