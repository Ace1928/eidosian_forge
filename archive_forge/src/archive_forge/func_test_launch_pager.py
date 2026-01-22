import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_launch_pager(self) -> None:
    ap = autopage.AutoPager()
    with mock.patch.object(ap, 'to_terminal', return_value=True), mock.patch.object(ap, '_paged_stream') as page, mock.patch.object(ap, '_reconfigure_output_stream') as reconf:
        with ap as stream:
            page.assert_called_once()
            self.assertIs(page.return_value, stream)
            reconf.assert_not_called()