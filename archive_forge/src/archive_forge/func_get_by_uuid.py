from troveclient import base
def get_by_uuid(self, datastore_version):
    """Get a specific datastore version.

        :rtype: :class:`DatastoreVersion`
        """
    return self._get('/datastores/versions/%s' % base.getid(datastore_version), 'version')