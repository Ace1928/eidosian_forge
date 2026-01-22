from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, Optional
import bson
from bson.binary import Binary
from bson.son import SON
from pymongo.errors import ConfigurationError, OperationFailure
def get_current_token(self, use_callback: bool=True) -> Optional[str]:
    properties = self.properties
    cb = properties.request_token_callback if use_callback else None
    cb_type = 'human'
    prev_token = self.access_token
    if prev_token:
        return prev_token
    if not use_callback and (not prev_token):
        return None
    if not prev_token and cb is not None:
        with self.lock:
            new_token = self.access_token
            if new_token != prev_token:
                return new_token
            if cb_type == 'human':
                context = {'timeout_seconds': CALLBACK_TIMEOUT_SECONDS, 'version': CALLBACK_VERSION, 'refresh_token': self.refresh_token}
                resp = cb(self.idp_info, context)
                self.validate_request_token_response(resp)
            self.token_gen_id += 1
    return self.access_token