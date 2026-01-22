import time
from paste.httpserver import WSGIServer
def reset_expires(self):
    if self.timeout:
        self.expires = time.time() + self.timeout