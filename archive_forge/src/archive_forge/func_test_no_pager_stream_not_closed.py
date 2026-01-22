import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_no_pager_stream_not_closed(self) -> None:
    flush = mock.MagicMock()
    with sinks.BufferFixture() as out:
        with autopage.AutoPager(out.stream) as stream:
            stream.flush = flush
            stream.write('foo')
        self.assertFalse(out.stream.closed)
    flush.assert_called_once()