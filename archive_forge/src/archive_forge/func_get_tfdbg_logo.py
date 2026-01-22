import math
import numpy as np
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import tensor_format
from tensorflow.python.debug.lib import common
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
def get_tfdbg_logo():
    """Make an ASCII representation of the tfdbg logo."""
    lines = ['', 'TTTTTT FFFF DDD  BBBB   GGG ', '  TT   F    D  D B   B G    ', '  TT   FFF  D  D BBBB  G  GG', '  TT   F    D  D B   B G   G', '  TT   F    DDD  BBBB   GGG ', '']
    return debugger_cli_common.RichTextLines(lines)