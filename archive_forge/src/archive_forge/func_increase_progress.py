from collections import OrderedDict
import numpy as np
import os
import re
import struct
import sys
import time
import logging
def increase_progress(self, extra_progress):
    """increase_progress(extra_progress)

        Increase the progress by a certain amount.
        """
    self.set_progress(self._progress + extra_progress)