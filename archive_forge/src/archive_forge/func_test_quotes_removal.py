from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_quotes_removal(self):
    self.run_script('\n$ echo \'cat\' "dog" \'"chicken"\' "\'dragon\'"\ncat dog "chicken" \'dragon\'\n')