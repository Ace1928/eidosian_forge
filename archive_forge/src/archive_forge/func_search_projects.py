from openstack.cloud import _utils
from openstack import exceptions
from openstack.identity.v3._proxy import Proxy
from openstack import utils
def search_projects(self, name_or_id=None, filters=None, domain_id=None):
    """Backwards compatibility method for search_projects

        search_projects originally had a parameter list that was name_or_id,
        filters and list had domain_id first. This method exists in this form
        to allow code written with positional parameter to still work. But
        really, use keyword arguments.

        :param name_or_id: Name or ID of the project(s).
        :param filters: dictionary of meta data to use for further filtering.
            Elements of this dictionary may, themselves, be dictionaries.
            Example::

                {
                  'last_name': 'Smith',
                  'other': {
                      'gender': 'Female'
                  }
                }

            OR

            A string containing a jmespath expression for further filtering.
            Example::

                "[?last_name==`Smith`] | [?other.gender]==`Female`]"
        :param domain_id: Domain ID to scope the searched projects.
        :returns: A list of identity ``Project`` objects.
        """
    projects = self.list_projects(domain_id=domain_id, filters=filters)
    return _utils._filter_list(projects, name_or_id, filters)