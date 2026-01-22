from functools import partial
from oslo_log import log
import stevedore
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import resource_options as ro
@classmethod
def check_auth_methods_against_rules(cls, user_id, auth_methods):
    """Validate the MFA rules against the successful auth methods.

        :param user_id: The user's ID (uuid).
        :type user_id: str
        :param auth_methods: List of methods that were used for auth
        :type auth_methods: set
        :returns: Boolean, ``True`` means rules match and auth may proceed,
                  ``False`` means rules do not match.
        """
    user_ref = PROVIDERS.identity_api.get_user(user_id)
    mfa_rules = user_ref['options'].get(ro.MFA_RULES_OPT.option_name, [])
    mfa_rules_enabled = user_ref['options'].get(ro.MFA_ENABLED_OPT.option_name, True)
    rules = cls._parse_rule_structure(mfa_rules, user_ref['id'])
    if not rules or not mfa_rules_enabled:
        LOG.debug('MFA Rules not processed for user `%(user_id)s`. Rule list: `%(rules)s` (Enabled: `%(enabled)s`).', {'user_id': user_id, 'rules': mfa_rules, 'enabled': mfa_rules_enabled})
        return True
    for r in rules:
        r_set = set(r).intersection(cls._auth_methods())
        if set(auth_methods).issuperset(r_set):
            LOG.debug('Auth methods for user `%(user_id)s`, `%(methods)s` matched MFA rule `%(rule)s`. Loaded auth_methods: `%(loaded)s`', {'user_id': user_id, 'rule': list(r_set), 'methods': auth_methods, 'loaded': cls._auth_methods()})
            return True
    LOG.debug('Auth methods for user `%(user_id)s`, `%(methods)s` did not match a MFA rule in `%(rules)s`.', {'user_id': user_id, 'methods': auth_methods, 'rules': rules})
    return False