import os
import sys
import stat
import select
import time
import errno
def split_command_line(command_line):
    """This splits a command line into a list of arguments. It splits arguments
    on spaces, but handles embedded quotes, doublequotes, and escaped
    characters. It's impossible to do this with a regular expression, so I
    wrote a little state machine to parse the command line. """
    arg_list = []
    arg = ''
    state_basic = 0
    state_esc = 1
    state_singlequote = 2
    state_doublequote = 3
    state_whitespace = 4
    state = state_basic
    for c in command_line:
        if state == state_basic or state == state_whitespace:
            if c == '\\':
                state = state_esc
            elif c == "'":
                state = state_singlequote
            elif c == '"':
                state = state_doublequote
            elif c.isspace():
                if state == state_whitespace:
                    None
                else:
                    arg_list.append(arg)
                    arg = ''
                    state = state_whitespace
            else:
                arg = arg + c
                state = state_basic
        elif state == state_esc:
            arg = arg + c
            state = state_basic
        elif state == state_singlequote:
            if c == "'":
                state = state_basic
            else:
                arg = arg + c
        elif state == state_doublequote:
            if c == '"':
                state = state_basic
            else:
                arg = arg + c
    if arg != '':
        arg_list.append(arg)
    return arg_list