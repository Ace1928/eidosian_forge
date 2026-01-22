import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
def decode_auth(auth: Union[str, bytes]) -> Tuple[str, str]:
    if isinstance(auth, str):
        auth = auth.encode('ascii')
    s = base64.b64decode(auth)
    login, pwd = s.split(b':', 1)
    return (login.decode('utf8'), pwd.decode('utf8'))