import collections
import copy
import json
import re
import shutil
import tempfile
def check_line_split(code_line):
    """Checks if a line was split with `\\`.

  Args:
      code_line: A line of Python code

  Returns:
    If the line was split with `\\`

  >>> skip_magic("!gcloud ml-engine models create ${MODEL} \\\\\\n")
  True
  """
    return re.search('\\\\\\s*\\n$', code_line)