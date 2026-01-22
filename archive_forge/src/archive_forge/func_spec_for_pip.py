import sys
import os
def spec_for_pip(self):
    """
        Ensure stdlib distutils when running under pip.
        See pypa/pip#8761 for rationale.
        """
    if sys.version_info >= (3, 12) or self.pip_imported_during_build():
        return
    clear_distutils()
    self.spec_for_distutils = lambda: None