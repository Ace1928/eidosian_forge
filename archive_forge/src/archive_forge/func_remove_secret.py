import base64
from .. import errors
from .. import utils
@utils.minimum_version('1.25')
@utils.check_resource('id')
def remove_secret(self, id):
    """
            Remove a secret

            Args:
                id (string): Full ID of the secret to remove

            Returns (boolean): True if successful

            Raises:
                :py:class:`docker.errors.NotFound`
                    if no secret with that ID exists
        """
    url = self._url('/secrets/{0}', id)
    res = self._delete(url)
    self._raise_for_status(res)
    return True