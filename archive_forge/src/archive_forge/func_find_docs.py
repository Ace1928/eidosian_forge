from __future__ import annotations
import collections.abc as c
import datetime
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .config import (
from . import junit_xml
def find_docs(self) -> t.Optional[str]:
    """Return the docs URL for this test or None if there is no docs URL."""
    if self.command != 'sanity':
        return None
    filename = f'{self.test}.html' if self.test else ''
    url = get_docs_url(f'https://docs.ansible.com/ansible-core/devel/dev_guide/testing/{self.command}/{filename}')
    return url