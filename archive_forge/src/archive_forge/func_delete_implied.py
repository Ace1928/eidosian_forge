from debtcollector import removals
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
@removals.remove(message='Use %s.delete instead.' % deprecation_msg, version='3.9.0', removal_version='4.0.0')
def delete_implied(self, prior_role, implied_role, **kwargs):
    return InferenceRuleManager(self.client).delete(prior_role, implied_role)