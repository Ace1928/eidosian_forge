from typing import cast, List, Type, Union, ValuesView
from .._connection import Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import CLIENT, CLOSED, DONE, MUST_CLOSE, SERVER
from .._util import Sentinel
def receive_and_get(conn: Connection, data: bytes) -> List[Event]:
    conn.receive_data(data)
    return get_all_events(conn)