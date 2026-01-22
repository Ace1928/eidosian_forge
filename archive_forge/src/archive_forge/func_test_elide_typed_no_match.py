import sys
import unittest
import os
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from IPython.testing import tools as tt
from IPython.terminal.ptutils import _elide, _adjust_completion_text_based_on_context
from IPython.terminal.shortcuts.auto_suggest import NavigableAutoSuggestFromHistory
def test_elide_typed_no_match(self):
    """
        if the match is too short we don't elide.
        avoid the "the...the"
        """
    self.assertEqual(_elide('the quick brown fox jumped over the lazy dog', 'the quick red fox', min_elide=10), 'the quick brown fox jumped over the lazy dog')