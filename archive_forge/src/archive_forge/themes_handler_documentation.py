from __future__ import annotations
import os
import re
from glob import glob
from typing import Any, Generator
from urllib.parse import urlparse
from jupyter_server.base.handlers import FileFindHandler
from jupyter_server.utils import url_path_join as ujoin
Replace the matched relative url with the mangled url.