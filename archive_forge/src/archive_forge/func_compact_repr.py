from __future__ import annotations
from ruamel.yaml.tag import Tag
def compact_repr(self) -> str:
    flow = ' {}' if self.flow_style else ''
    anchor = f' &{self.anchor}' if self.anchor else ''
    tag = f' <{self.tag!s}>' if self.tag else ''
    return f'{self.crepr}{flow}{anchor}{tag}'