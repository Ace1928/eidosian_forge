import contextlib
import functools
import hashlib
import os
import re
import sys
import textwrap
from argparse import Namespace
from dataclasses import fields, is_dataclass
from enum import auto, Enum
from typing import (
from typing_extensions import Self
from torchgen.code_template import CodeTemplate
def write_with_template(self, filename: str, template_fn: str, env_callable: Callable[[], Union[str, Dict[str, Any]]]) -> None:
    filename = f'{self.install_dir}/{filename}'
    assert filename not in self.filenames, 'duplicate file write {filename}'
    self.filenames.add(filename)
    if not self.dry_run:
        substitute_out = self.substitute_with_template(template_fn=template_fn, env_callable=env_callable)
        self._write_if_changed(filename=filename, contents=substitute_out)