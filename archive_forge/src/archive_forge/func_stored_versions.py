from pprint import pformat
from six import iteritems
import re
@stored_versions.setter
def stored_versions(self, stored_versions):
    """
        Sets the stored_versions of this V1beta1CustomResourceDefinitionStatus.
        StoredVersions are all versions of CustomResources that were ever
        persisted. Tracking these versions allows a migration path for stored
        versions in etcd. The field is mutable so the migration controller can
        first finish a migration to another version (i.e. that no old objects
        are left in the storage), and then remove the rest of the versions from
        this list. None of the versions in this list can be removed from the
        spec.Versions field.

        :param stored_versions: The stored_versions of this
        V1beta1CustomResourceDefinitionStatus.
        :type: list[str]
        """
    if stored_versions is None:
        raise ValueError('Invalid value for `stored_versions`, must not be `None`')
    self._stored_versions = stored_versions