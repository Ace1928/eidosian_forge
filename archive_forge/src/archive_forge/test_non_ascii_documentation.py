import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
Stop the test if it's about to fail or errors out.

        Until we get proper support on OSX for accented paths (in fact, any
        path whose NFD decomposition is different than the NFC one), this is
        the best way to keep test active (as opposed to disabling them
        completely). This is a stop gap. The tests should at least be rewritten
        so that the failing ones are clearly separated from the passing ones.
        