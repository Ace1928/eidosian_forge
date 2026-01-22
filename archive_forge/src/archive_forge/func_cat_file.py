import base64
import io
import re
import requests
import fsspec
def cat_file(self, path, start=None, end=None, **kwargs):
    path = self._strip_protocol(path)
    r = self.session.get(f'{self.url}/{path}')
    if r.status_code == 404:
        return FileNotFoundError(path)
    r.raise_for_status()
    out = r.json()
    if out['format'] == 'text':
        b = out['content'].encode()
    else:
        b = base64.b64decode(out['content'])
    return b[start:end]