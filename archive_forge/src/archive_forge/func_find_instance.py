from openstack.database.v1 import database as _database
from openstack.database.v1 import flavor as _flavor
from openstack.database.v1 import instance as _instance
from openstack.database.v1 import user as _user
from openstack import proxy
def find_instance(self, name_or_id, ignore_missing=True):
    """Find a single instance

        :param name_or_id: The name or ID of a instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the resource does not exist.
            When set to ``True``, None will be returned when
            attempting to find a nonexistent resource.
        :returns: One :class:`~openstack.database.v1.instance.Instance` or None
        """
    return self._find(_instance.Instance, name_or_id, ignore_missing=ignore_missing)