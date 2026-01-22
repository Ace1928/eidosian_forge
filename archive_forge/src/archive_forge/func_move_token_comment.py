from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.scanner import Scanner, RoundTripScanner, ScannerError  # NOQA
from ruamel.yaml.scanner import BlankLineComment
from ruamel.yaml.comments import C_PRE, C_POST, C_SPLIT_ON_FIRST_BLANK
from ruamel.yaml.compat import nprint, nprintf  # NOQA
from ruamel.yaml.tag import Tag
def move_token_comment(self: Any, token: Any, nt: Any=None, empty: Optional[bool]=False) -> None:
    token.move_new_comment(self.scanner.peek_token() if nt is None else nt, empty=empty)