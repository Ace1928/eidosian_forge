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
def get_next_sequence_number(self):
    seq = self.obj_dict['current_child_sequence']
    self.obj_dict['current_child_sequence'] += 1
    return seq