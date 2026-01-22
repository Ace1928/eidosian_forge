from .. import errors
from .. import utils
from ..types import CancellableStream
def exec_inspect(self, exec_id):
    """
        Return low-level information about an exec command.

        Args:
            exec_id (str): ID of the exec instance

        Returns:
            (dict): Dictionary of values returned by the endpoint.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    if isinstance(exec_id, dict):
        exec_id = exec_id.get('Id')
    res = self._get(self._url('/exec/{0}/json', exec_id))
    return self._result(res, True)