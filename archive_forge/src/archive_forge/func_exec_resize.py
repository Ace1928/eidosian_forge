from .. import errors
from .. import utils
from ..types import CancellableStream
def exec_resize(self, exec_id, height=None, width=None):
    """
        Resize the tty session used by the specified exec command.

        Args:
            exec_id (str): ID of the exec instance
            height (int): Height of tty session
            width (int): Width of tty session
        """
    if isinstance(exec_id, dict):
        exec_id = exec_id.get('Id')
    params = {'h': height, 'w': width}
    url = self._url('/exec/{0}/resize', exec_id)
    res = self._post(url, params=params)
    self._raise_for_status(res)