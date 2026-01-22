import requests
from ..spec import AbstractFileSystem
from ..utils import infer_storage_options
from .memory import MemoryFile
@property
def branches(self):
    """Names of branches in the repo"""
    r = requests.get(f'https://api.github.com/repos/{self.org}/{self.repo}/branches', timeout=self.timeout, **self.kw)
    r.raise_for_status()
    return [t['name'] for t in r.json()]