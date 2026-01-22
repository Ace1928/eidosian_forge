from __future__ import absolute_import
import cython
from . import Errors
from .Regexps import BOL, EOL, EOF
def next_char(self):
    input_state = self.input_state
    if self.trace:
        print('Scanner: next: %s [%d] %d' % (' ' * 20, input_state, self.cur_pos))
    if input_state == 1:
        self.cur_pos = self.next_pos
        c = self.read_char()
        if c == u'\n':
            self.cur_char = EOL
            self.input_state = 2
        elif not c:
            self.cur_char = EOL
            self.input_state = 4
        else:
            self.cur_char = c
    elif input_state == 2:
        self.cur_char = u'\n'
        self.input_state = 3
    elif input_state == 3:
        self.cur_line += 1
        self.cur_line_start = self.cur_pos = self.next_pos
        self.cur_char = BOL
        self.input_state = 1
    elif input_state == 4:
        self.cur_char = EOF
        self.input_state = 5
    else:
        self.cur_char = u''
    if self.trace:
        print('--> [%d] %d %r' % (input_state, self.cur_pos, self.cur_char))