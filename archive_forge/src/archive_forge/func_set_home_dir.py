import ast
import os
import sys
import tokenize
import types
from inspect import CO_COROUTINE
from gradio.wasm_utils import app_id_context
def set_home_dir(home_dir: str) -> None:
    os.environ['HOME'] = home_dir
    os.chdir(home_dir)