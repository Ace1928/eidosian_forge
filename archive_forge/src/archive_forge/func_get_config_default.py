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
def get_config_default(self, group=None, option=None):
    """Get default config, or partial default config.

        :param group: an optional specific group of options
        :param option: an optional specific option within the group

        :returns: a dict of group dicts containing the default options,
                  filtered by group and option if specified
        :raises keystone.exception.InvalidDomainConfig: when the config
                and group/option parameters specify an option we do not
                support (or one that is not whitelisted).

        An example response::

            {
                'ldap': {
                    'url': 'myurl',
                    'user_tree_dn': 'OU=myou',
                    ....},
                'identity': {
                    'driver': 'ldap'}

            }

        """
    self._assert_valid_group_and_option(group, option)
    config_list = []
    if group:
        if option:
            if option not in self.whitelisted_options[group]:
                msg = _('Reading the default for option %(option)s in group %(group)s is not supported') % {'option': option, 'group': group}
                raise exception.InvalidDomainConfig(reason=msg)
            config_list.append(self._option_dict(group, option))
        else:
            for each_option in self.whitelisted_options[group]:
                config_list.append(self._option_dict(group, each_option))
    else:
        for each_group in self.whitelisted_options:
            for each_option in self.whitelisted_options[each_group]:
                config_list.append(self._option_dict(each_group, each_option))
    return self._list_to_config(config_list, req_option=option)