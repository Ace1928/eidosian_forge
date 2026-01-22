from __future__ import annotations
from typing import TYPE_CHECKING, Container, Iterable, NoReturn
import attrs
import pytest
from ... import _abc, _core
from .tutil import check_sequence_matches
def test_null_instrument() -> None:

    class NullInstrument(_abc.Instrument):

        def something_unrelated(self) -> None:
            pass

    async def main() -> None:
        await _core.checkpoint()
    _core.run(main, instruments=[NullInstrument()])