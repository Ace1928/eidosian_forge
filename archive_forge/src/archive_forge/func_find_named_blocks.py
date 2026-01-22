import datetime
from io import StringIO
import linecache
import os.path
import posixpath
import re
import threading
from tornado import escape
from tornado.log import app_log
from tornado.util import ObjectDict, exec_in, unicode_type
from typing import Any, Union, Callable, List, Dict, Iterable, Optional, TextIO
import typing
def find_named_blocks(self, loader: Optional[BaseLoader], named_blocks: Dict[str, _NamedBlock]) -> None:
    assert loader is not None
    included = loader.load(self.name, self.template_name)
    included.file.find_named_blocks(loader, named_blocks)