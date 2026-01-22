import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def get_simplify(self):
    """Get whether to simplify or not.

        Refer to set_simplify for more information.
        """
    return self.obj_dict['simplify']