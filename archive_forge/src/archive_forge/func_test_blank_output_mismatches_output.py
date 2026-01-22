from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_blank_output_mismatches_output(self):
    """If you give output, the output must actually be blank.

        See <https://bugs.launchpad.net/bzr/+bug/637830>: previously blank
        output was a wildcard.  Now you must say ... if you want that.
        """
    self.assertRaises(AssertionError, self.run_script, '\n            $ echo foo\n            ')