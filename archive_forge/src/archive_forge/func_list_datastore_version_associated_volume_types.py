from troveclient import base
def list_datastore_version_associated_volume_types(self, datastore, version_id):
    """Get a list of all volume-types for the specified datastore type
        and datastore version .
        :rtype: list of :class:`VolumeType`.
        """
    return self._list('/datastores/%s/versions/%s/volume-types' % (datastore, version_id), 'volume_types')