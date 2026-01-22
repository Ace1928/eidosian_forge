from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
from urllib.parse import urlparse
def url_project(self) -> str:
    assert self.project
    return f'{self.url_entity()}/{self.project}'