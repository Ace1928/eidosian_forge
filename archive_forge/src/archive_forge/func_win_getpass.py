import contextlib
import io
import os
import sys
import warnings
def win_getpass(prompt='Password: ', stream=None):
    """Prompt for password with echo off, using Windows getwch()."""
    if sys.stdin is not sys.__stdin__:
        return fallback_getpass(prompt, stream)
    for c in prompt:
        msvcrt.putwch(c)
    pw = ''
    while 1:
        c = msvcrt.getwch()
        if c == '\r' or c == '\n':
            break
        if c == '\x03':
            raise KeyboardInterrupt
        if c == '\x08':
            pw = pw[:-1]
        else:
            pw = pw + c
    msvcrt.putwch('\r')
    msvcrt.putwch('\n')
    return pw