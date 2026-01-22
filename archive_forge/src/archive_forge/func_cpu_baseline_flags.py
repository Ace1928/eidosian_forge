import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def cpu_baseline_flags(self):
    """
        Returns a list of final CPU baseline compiler flags
        """
    return self.parse_baseline_flags