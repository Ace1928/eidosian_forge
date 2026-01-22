import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def handle_block_comment(self, current_token, preserve_statement_flags):
    if self._output.raw:
        self._output.add_raw_token(current_token)
        if current_token.directives and current_token.directives.get('preserve') == 'end':
            self._output.raw = self._options.test_output_raw
        return
    if current_token.directives:
        self.print_newline(preserve_statement_flags=preserve_statement_flags)
        self.print_token(current_token)
        if current_token.directives.get('preserve') == 'start':
            self._output.raw = True
        self.print_newline(preserve_statement_flags=True)
        return
    if not self.acorn.newline.search(current_token.text) and (not current_token.newlines):
        self._output.space_before_token = True
        self.print_token(current_token)
        self._output.space_before_token = True
        return
    lines = self.acorn.allLineBreaks.split(current_token.text)
    javadoc = False
    starless = False
    last_indent = current_token.whitespace_before
    last_indent_length = len(last_indent)
    self.print_newline(preserve_statement_flags=preserve_statement_flags)
    self.print_token(current_token, lines[0])
    self.print_newline(preserve_statement_flags=preserve_statement_flags)
    if len(lines) > 1:
        lines = lines[1:]
        javadoc = not any((l for l in lines if l.strip() == '' or l.lstrip()[0] != '*'))
        starless = all((l.startswith(last_indent) or l.strip() == '' for l in lines))
        if javadoc:
            self._flags.alignment = 1
        for line in lines:
            if javadoc:
                self.print_token(current_token, line.lstrip())
            elif starless and len(line) > last_indent_length:
                self.print_token(current_token, line[last_indent_length:])
            else:
                self._output.current_line.set_indent(-1)
                self._output.add_token(line)
            self.print_newline(preserve_statement_flags=preserve_statement_flags)
        self._flags.alignment = 0