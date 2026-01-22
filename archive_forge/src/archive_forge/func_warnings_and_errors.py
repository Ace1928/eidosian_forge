import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
@property
def warnings_and_errors(self):
    return [log for log in self._log if log[0] in (WARNING, ERROR)]