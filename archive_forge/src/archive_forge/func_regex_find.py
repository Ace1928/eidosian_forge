import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def regex_find(orig_screen_output, regex, font_attr):
    """Perform regex match in rich text lines.

  Produces a new RichTextLines object with font_attr_segs containing highlighted
  regex matches.

  Example use cases include:
  1) search for specific items in a large list of items, and
  2) search for specific numerical values in a large tensor.

  Args:
    orig_screen_output: The original RichTextLines, in which the regex find
      is to be performed.
    regex: The regex used for matching.
    font_attr: Font attribute used for highlighting the found result.

  Returns:
    A modified copy of orig_screen_output.

  Raises:
    ValueError: If input str regex is not a valid regular expression.
  """
    new_screen_output = RichTextLines(orig_screen_output.lines, font_attr_segs=copy.deepcopy(orig_screen_output.font_attr_segs), annotations=orig_screen_output.annotations)
    try:
        re_prog = re.compile(regex)
    except re.error:
        raise ValueError('Invalid regular expression: "%s"' % regex)
    regex_match_lines = []
    for i, line in enumerate(new_screen_output.lines):
        find_it = re_prog.finditer(line)
        match_segs = []
        for match in find_it:
            match_segs.append((match.start(), match.end(), font_attr))
        if match_segs:
            if i not in new_screen_output.font_attr_segs:
                new_screen_output.font_attr_segs[i] = match_segs
            else:
                new_screen_output.font_attr_segs[i].extend(match_segs)
                new_screen_output.font_attr_segs[i] = sorted(new_screen_output.font_attr_segs[i], key=lambda x: x[0])
            regex_match_lines.append(i)
    new_screen_output.annotations[REGEX_MATCH_LINES_KEY] = regex_match_lines
    return new_screen_output