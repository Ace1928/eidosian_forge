from __future__ import annotations
import json
import urllib.parse as urlparse
from typing import (
import param
from ..models.location import Location as _BkLocation
from ..reactive import Syncable
from ..util import edit_readonly, parse_query
from .document import create_doc_if_none_exists
from .state import state
def unsync(self, parameterized: param.Parameterized, parameters: Optional[List[str]]=None) -> None:
    """
        Unsyncs the parameters of the Parameterized with the query
        params in the URL. If no parameters are supplied all
        parameters except the name are unsynced.

        Arguments
        ---------
        parameterized (param.Parameterized):
          The Parameterized object to unsync query parameters with
        parameters (list):
          A list of parameters to unsync.
        """
    matches = [s for s in self._synced if s[0] is parameterized]
    if not matches:
        ptype = type(parameterized)
        raise ValueError(f'Cannot unsync {ptype} object since it was never synced in the first place.')
    synced, unsynced = ([], [])
    for p, params, watcher, on_error in self._synced:
        if parameterized is not p:
            synced.append((p, params, watcher, on_error))
            continue
        parameterized.param.unwatch(watcher)
        if parameters is None:
            unsynced.extend(list(params.values()))
        else:
            unsynced.extend([q for p, q in params.items() if p in parameters])
            new_params = {p: q for p, q in params.items() if p not in parameters}
            new_watcher = parameterized.param.watch(watcher.fn, list(new_params))
            synced.append((p, new_params, new_watcher, on_error))
    self._synced = synced
    query = {k: v for k, v in self.query_params.items() if k not in unsynced}
    self.search = '?' + urlparse.urlencode(query) if query else ''