from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
def reset_view_state(self):
    """
        Resets view state.
        """
    self.view_state = self.raytracing_data.initial_view_state()