import requests
from ..spec import AbstractFileSystem
from ..utils import infer_storage_options
from .memory import MemoryFile
@property
def kw(self):
    if self.username:
        return {'auth': (self.username, self.token)}
    return {}