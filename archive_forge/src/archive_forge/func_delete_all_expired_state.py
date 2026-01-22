from __future__ import annotations
import datetime
import os
import threading
from collections import OrderedDict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterator
def delete_all_expired_state(self):
    for session_id in self.session_data:
        self.delete_state(session_id, expired_only=True)