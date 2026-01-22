import unittest
from nltk import tgrep
from nltk.tree import ParentedTree

        Test that semicolons at the end of a tgrep2 search string won't
        cause a parse failure.
        