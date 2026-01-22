from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def prefilter_line_info(self, line_info):
    """Prefilter a line that has been converted to a LineInfo object.

        This implements the checker/handler part of the prefilter pipe.
        """
    handler = self.find_handler(line_info)
    return handler.handle(line_info)