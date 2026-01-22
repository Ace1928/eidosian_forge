from openstack.database.v1 import database as _database
from openstack.database.v1 import flavor as _flavor
from openstack.database.v1 import instance as _instance
from openstack.database.v1 import user as _user
from openstack import proxy
def find_database(self, name_or_id, instance, ignore_missing=True):
    """Find a single database

        :param name_or_id: The name or ID of a database.
        :param instance: This can be either the ID of an instance
            or a :class:`~openstack.database.v1.instance.Instance`
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the resource does not exist.
            When set to ``True``, None will be returned when
            attempting to find a nonexistent resource.
        :returns: One :class:`~openstack.database.v1.database.Database` or None
        """
    instance = self._get_resource(_instance.Instance, instance)
    return self._find(_database.Database, name_or_id, instance_id=instance.id, ignore_missing=ignore_missing)