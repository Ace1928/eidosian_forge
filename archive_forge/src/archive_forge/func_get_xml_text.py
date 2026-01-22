import abc
import hashlib
import json
import xml.etree.ElementTree as ET  # noqa
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Optional, Tuple, Union
from jupyter_server.base.handlers import APIHandler
from jupyterlab_server.translation_utils import translator
from packaging.version import parse
from tornado import httpclient, web
from jupyterlab._version import __version__
def get_xml_text(attr: str, default: Optional[str]=None) -> str:
    node_item = node.find(f'atom:{attr}', xml_namespaces)
    if node_item is not None:
        return node_item.text
    elif default is not None:
        return default
    else:
        error_m = f'atom feed entry does not contain a required attribute: {attr}'
        raise KeyError(error_m)