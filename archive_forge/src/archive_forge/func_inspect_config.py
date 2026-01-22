import base64
from .. import utils
@utils.minimum_version('1.30')
@utils.check_resource('id')
def inspect_config(self, id):
    """
            Retrieve config metadata

            Args:
                id (string): Full ID of the config to inspect

            Returns (dict): A dictionary of metadata

            Raises:
                :py:class:`docker.errors.NotFound`
                    if no config with that ID exists
        """
    url = self._url('/configs/{0}', id)
    return self._result(self._get(url), True)