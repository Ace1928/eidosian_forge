from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import rmtree
from typing import List
from tornado.concurrent import run_on_executor
from tornado.gen import convert_yielded
from .manager import lsp_message_listener
from .paths import file_uri_to_path, is_relative
from .types import LanguageServerManagerAPI
@property
def full_range(self):
    start = {'line': 0, 'character': 0}
    end = {'line': len(self.lines), 'character': len(self.lines[-1]) if self.lines else 0}
    return {'start': start, 'end': end}