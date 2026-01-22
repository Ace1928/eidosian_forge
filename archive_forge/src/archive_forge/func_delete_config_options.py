import abc
from keystone import exception
@abc.abstractmethod
def delete_config_options(self, domain_id, group=None, option=None):
    """Delete config options for a domain.

        Allows deletion of all options for a domain, all options in a group
        or a specific option. The driver is silent if there are no options
        to delete.

        :param domain_id: the domain for this option
        :param group: optional group option name
        :param option: optional option name. If group is None, then this
                       parameter is ignored

        The option is uniquely defined by domain_id, group and option,
        irrespective of whether it is sensitive ot not.

        """
    raise exception.NotImplemented()