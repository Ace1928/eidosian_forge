import sys
import os
def spec_for_sensitive_tests(self):
    """
        Ensure stdlib distutils when running select tests under CPython.

        python/cpython#91169
        """
    clear_distutils()
    self.spec_for_distutils = lambda: None