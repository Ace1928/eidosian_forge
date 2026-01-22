import copy
import logging
import threading
import google.auth.exceptions as e
def start_refresh(self, cred, request):
    """Starts a refresh thread for the given credentials.
        The credentials are refreshed using the request parameter.
        request and cred MUST not be None

        Returns True if a background refresh was kicked off. False otherwise.

        Args:
            cred: A credentials object.
            request: A request object.
        Returns:
          bool
        """
    if cred is None or request is None:
        raise e.InvalidValue('Unable to start refresh. cred and request must be valid and instantiated objects.')
    with self._lock:
        if self._worker is not None and self._worker._error_info is not None:
            return False
        if self._worker is None or not self._worker.is_alive():
            self._worker = RefreshThread(cred=cred, request=copy.deepcopy(request))
            self._worker.start()
    return True