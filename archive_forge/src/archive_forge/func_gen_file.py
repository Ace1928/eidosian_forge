import json
import os
import sys
def gen_file(name, contents):
    """Generate the file.

    This writes the file to be generated back to the controller.

    Args:
        name: (str) The UNIX-style relative path of the file.
        contents: (str) The complete file contents.
    """
    _write_msg(type='gen_file', filename=name, contents=contents)