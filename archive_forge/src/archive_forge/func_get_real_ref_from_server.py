from contextlib import contextmanager
from abc import ABC
from abc import abstractmethod
def get_real_ref_from_server(self):
    global _current_server
    if _current_server is None:
        return None
    client_map = _current_server.client_side_ref_map.get(self.client_id, None)
    if client_map is None:
        return None
    return client_map.get(self.id, None)