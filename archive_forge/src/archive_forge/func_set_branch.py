import time
from io import BytesIO
from ... import errors as bzr_errors
from ... import tests
from ...tests.features import Feature, ModuleAvailableFeature
from .. import import_dulwich
def set_branch(self, branch):
    """Set the branch we are committing."""
    self._branch = branch