import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def search_volume_types(self, name_or_id=None, filters=None, get_extra=None):
    """Search for one or more volume types.

        :param name_or_id: Name or unique ID of volume type(s).
        :param filters: **DEPRECATED** A dictionary of meta data to use for
            further filtering. Elements of this dictionary may, themselves, be
            dictionaries. Example::

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

        :returns: A list of volume ``Type`` objects, if any are found.
        """
    volume_types = self.list_volume_types(get_extra=get_extra)
    return _utils._filter_list(volume_types, name_or_id, filters)