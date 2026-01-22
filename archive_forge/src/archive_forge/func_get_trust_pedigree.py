from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def get_trust_pedigree(self, trust_id):
    trust = self.driver.get_trust(trust_id)
    trust_chain = [trust]
    while trust and trust.get('redelegated_trust_id'):
        trust = self.driver.get_trust(trust['redelegated_trust_id'])
        trust_chain.append(trust)
    return trust_chain