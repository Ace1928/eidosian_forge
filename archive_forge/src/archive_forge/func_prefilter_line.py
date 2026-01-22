from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def prefilter_line(self, line, continue_prompt=False):
    """Prefilter a single input line as text.

        This method prefilters a single line of text by calling the
        transformers and then the checkers/handlers.
        """
    self.shell._last_input_line = line
    if not line:
        return ''
    if not continue_prompt or (continue_prompt and self.multi_line_specials):
        line = self.transform_line(line, continue_prompt)
    line_info = LineInfo(line, continue_prompt)
    stripped = line.strip()
    normal_handler = self.get_handler_by_name('normal')
    if not stripped:
        return normal_handler.handle(line_info)
    if continue_prompt and (not self.multi_line_specials):
        return normal_handler.handle(line_info)
    prefiltered = self.prefilter_line_info(line_info)
    return prefiltered