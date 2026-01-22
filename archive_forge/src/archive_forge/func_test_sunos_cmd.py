import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_sunos_cmd(self) -> None:
    with PlatformFixture('sunos5'):
        cmd = command.PlatformPager()
        self.assertEqual(['less'], cmd.command())