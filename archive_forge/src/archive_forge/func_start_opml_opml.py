from __future__ import annotations
import copy
from . import common, dates
def start_opml_opml(self, attrs: dict[str, str]) -> None:
    self.harvest['version'] = 'opml'
    if attrs.get('version') in ('1.0', '1.1'):
        self.harvest['version'] = 'opml1'
    elif attrs.get('version') == '2.0':
        self.harvest['version'] = 'opml2'