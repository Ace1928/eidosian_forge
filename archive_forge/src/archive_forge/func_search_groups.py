from openstack.cloud import _utils
from openstack import exceptions
from openstack.identity.v3._proxy import Proxy
from openstack import utils
@_utils.valid_kwargs('domain_id')
def search_groups(self, name_or_id=None, filters=None, **kwargs):
    """Search Keystone groups.

        :param name_or_id: Name or ID of the group(s).
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
        :param domain_id: domain id.

        :returns: A list of identity ``Group`` objects
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call.
        """
    groups = self.list_groups(**kwargs)
    return _utils._filter_list(groups, name_or_id, filters)