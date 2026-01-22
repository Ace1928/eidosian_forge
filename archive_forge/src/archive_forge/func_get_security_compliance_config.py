from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def get_security_compliance_config(self, domain_id, group, option=None):
    """Get full or partial security compliance config from configuration.

        :param domain_id: the domain in question
        :param group: a specific group of options
        :param option: an optional specific option within the group

        :returns: a dict of group dicts containing the whitelisted options,
                  filtered by group and option specified
        :raises keystone.exception.InvalidDomainConfig: when the config
                and group/option parameters specify an option we do not
                support

        An example response::

            {
                'security_compliance': {
                    'password_regex': '^(?=.*\\d)(?=.*[a-zA-Z]).{7,}$'
                    'password_regex_description':
                        'A password must consist of at least 1 letter, '
                        '1 digit, and have a minimum length of 7 characters'
                    }
            }

        """
    if domain_id != CONF.identity.default_domain_id:
        msg = _('Reading security compliance information for any domain other than the default domain is not allowed or supported.')
        raise exception.InvalidDomainConfig(reason=msg)
    config_list = []
    readable_options = ['password_regex', 'password_regex_description']
    if option and option not in readable_options:
        msg = _('Reading security compliance values other than password_regex and password_regex_description is not allowed.')
        raise exception.InvalidDomainConfig(reason=msg)
    elif option and option in readable_options:
        config_list.append(self._option_dict(group, option))
    elif not option:
        for op in readable_options:
            config_list.append(self._option_dict(group, op))
    return self._list_to_config(config_list, req_option=option)