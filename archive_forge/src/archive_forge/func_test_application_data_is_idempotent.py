import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
def test_application_data_is_idempotent(self):
    """
        application data is idempotent

        """
    self.test_application_data()
    self.test_application_data()