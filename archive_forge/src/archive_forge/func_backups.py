from openstack.block_storage import _base_proxy
from openstack.block_storage.v2 import backup as _backup
from openstack.block_storage.v2 import capabilities as _capabilities
from openstack.block_storage.v2 import extension as _extension
from openstack.block_storage.v2 import limits as _limits
from openstack.block_storage.v2 import quota_set as _quota_set
from openstack.block_storage.v2 import snapshot as _snapshot
from openstack.block_storage.v2 import stats as _stats
from openstack.block_storage.v2 import type as _type
from openstack.block_storage.v2 import volume as _volume
from openstack.identity.v3 import project as _project
from openstack import resource
def backups(self, details=True, **query):
    """Retrieve a generator of backups

        :param bool details: When set to ``False`` no additional details will
            be returned. The default, ``True``, will cause objects with
            additional attributes to be returned.
        :param dict query: Optional query parameters to be sent to limit the
            resources being returned:

            * offset: pagination marker
            * limit: pagination limit
            * sort_key: Sorts by an attribute. A valid value is
              name, status, container_format, disk_format, size, id,
              created_at, or updated_at. Default is created_at.
              The API uses the natural sorting direction of the
              sort_key attribute value.
            * sort_dir: Sorts by one or more sets of attribute and sort
              direction combinations. If you omit the sort direction
              in a set, default is desc.

        :returns: A generator of backup objects.
        """
    base_path = '/backups/detail' if details else None
    return self._list(_backup.Backup, base_path=base_path, **query)