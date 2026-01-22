from .. import errors
from .. import utils
def remove_volume(self, name, force=False):
    """
        Remove a volume. Similar to the ``docker volume rm`` command.

        Args:
            name (str): The volume's name
            force (bool): Force removal of volumes that were already removed
                out of band by the volume driver plugin.

        Raises:
            :py:class:`docker.errors.APIError`
                If volume failed to remove.
        """
    params = {}
    if force:
        if utils.version_lt(self._version, '1.25'):
            raise errors.InvalidVersion('force removal was introduced in API 1.25')
        params = {'force': force}
    url = self._url('/volumes/{0}', name, params=params)
    resp = self._delete(url)
    self._raise_for_status(resp)