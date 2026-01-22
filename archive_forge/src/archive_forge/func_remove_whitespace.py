import os
from pathlib import Path
from unittest import mock
import cirq_web
def remove_whitespace(string: str) -> str:
    return ''.join(string.split())