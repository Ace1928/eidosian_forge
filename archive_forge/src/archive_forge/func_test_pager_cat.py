import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_pager_cat(self) -> None:
    outstream = mock.Mock()
    cat = command.CustomPager('cat')
    ap = autopage.AutoPager(outstream, pager_command=cat)
    with mock.patch.object(ap, 'to_terminal', return_value=True), mock.patch.object(ap, '_paged_stream') as page, mock.patch.object(ap, '_reconfigure_output_stream') as reconf:
        with ap as stream:
            page.assert_not_called()
            self.assertIs(outstream, stream)
            reconf.assert_called_once()