import json
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from wandb import util
from wandb.apis import InternalApi
def query_with_timeout(self, timeout: Union[int, float, None]=None) -> None:
    if self._settings and self._settings._disable_viewer:
        return
    timeout = timeout or 5
    async_viewer = util.async_call(self._api.viewer_server_info, timeout=timeout)
    try:
        viewer_tuple, viewer_thread = async_viewer()
    except Exception:
        self._error_network = True
        return
    if viewer_thread.is_alive():
        self._error_network = True
        return
    self._error_network = False
    self._viewer, self._serverinfo = viewer_tuple
    self._flags = json.loads(self._viewer.get('flags', '{}'))