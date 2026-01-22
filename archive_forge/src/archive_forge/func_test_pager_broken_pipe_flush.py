import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_pager_broken_pipe_flush(self) -> None:
    flush = mock.MagicMock(side_effect=BrokenPipeError)
    with sinks.TTYFixture() as out:
        ap = autopage.AutoPager(out.stream)
        with fixtures.MockPatch('subprocess.Popen') as popen:
            with sinks.BufferFixture() as pager_in:
                popen.mock.return_value.stdin = pager_in.stream
                with ap as stream:
                    stream.write('foo')
                    stream.close = flush
        self.assertEqual(141, ap.exit_code())