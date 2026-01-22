from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def getTagStarts(self) -> list[tuple[Literal['start'], str, dict[str, str]]]:
    return [token for token in self.tokens if token[0] == 'start']