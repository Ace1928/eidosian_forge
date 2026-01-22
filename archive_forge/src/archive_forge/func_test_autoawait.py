from itertools import chain, repeat
from textwrap import dedent, indent
from typing import TYPE_CHECKING
from unittest import TestCase
import pytest
from IPython.core.async_helpers import _should_be_async
from IPython.testing.decorators import skip_without
def test_autoawait(self):
    iprc('%autoawait False')
    iprc('%autoawait True')
    iprc('\n        from asyncio import sleep\n        await sleep(0.1)\n        ')