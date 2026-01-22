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
def delete_config(self, domain_id, group=None, option=None):
    """Delete config, or partial config, for the domain.

        :param domain_id: the domain in question
        :param group: an optional specific group of options
        :param option: an optional specific option within the group

        If group and option are None, then the entire config for the domain
        is deleted. If group is not None, then just that group of options will
        be deleted. If group and option are both specified, then just that
        option is deleted.

        :raises keystone.exception.InvalidDomainConfig: when group/option
                parameters specify an option we do not support or one that
                does not exist in the original config.

        """
    self._assert_valid_group_and_option(group, option)
    if group:
        current_config = self._get_config_with_sensitive_info(domain_id)
        current_group = current_config.get(group)
        if not current_group:
            msg = _('group %(group)s') % {'group': group}
            raise exception.DomainConfigNotFound(domain_id=domain_id, group_or_option=msg)
        if option and (not current_group.get(option)):
            msg = _('option %(option)s in group %(group)s') % {'group': group, 'option': option}
            raise exception.DomainConfigNotFound(domain_id=domain_id, group_or_option=msg)
    self.delete_config_options(domain_id, group, option)
    self.get_config_with_sensitive_info.invalidate(self, domain_id)