import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
def store_in_name(self, store_name):
    for_token = _Token(self.i_line, None, 'for ')
    self.tokens.append(for_token)
    prev = for_token
    t_name = _Token(store_name.i_line, store_name.instruction, after=prev)
    self.tokens.append(t_name)
    prev = t_name
    in_token = _Token(store_name.i_line, None, ' in ', after=prev)
    self.tokens.append(in_token)
    prev = in_token
    max_line = store_name.i_line
    if self.iter_in:
        for t in self.iter_in.tokens:
            t.mark_after(prev)
            max_line = max(max_line, t.i_line)
            prev = t
        self.tokens.extend(self.iter_in.tokens)
    colon_token = _Token(self.i_line, None, ':', after=prev)
    self.tokens.append(colon_token)
    prev = for_token
    self._write_tokens()