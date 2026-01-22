import base64
import os.path
import uuid
from .. import __version__
def gen_cookie_secret():
    return base64.b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes)