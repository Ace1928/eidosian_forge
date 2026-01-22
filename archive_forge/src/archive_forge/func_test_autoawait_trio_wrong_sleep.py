from itertools import chain, repeat
from textwrap import dedent, indent
from typing import TYPE_CHECKING
from unittest import TestCase
import pytest
from IPython.core.async_helpers import _should_be_async
from IPython.testing.decorators import skip_without
@skip_without('trio')
def test_autoawait_trio_wrong_sleep(self):
    iprc('%autoawait trio')
    res = iprc_nr('\n        import asyncio\n        await asyncio.sleep(0)\n        ')
    with self.assertRaises(TypeError):
        res.raise_error()