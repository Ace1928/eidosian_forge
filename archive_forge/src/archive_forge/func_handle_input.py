import sys
import os
import time
import locale
import signal
import urwid
from typing import Optional
from . import args as bpargs, repl, translations
from .formatter import theme_map
from .translations import _
from .keys import urwid_key_dispatch as key_dispatch
def handle_input(self, event):
    if self.frame.get_focus() != 'body':
        return
    if event == 'enter':
        inp = self.edit.get_edit_text()
        self.history.append(inp)
        self.edit.make_readonly()
        self.stdout_hist += inp
        self.stdout_hist += '\n'
        self.edit = None
        self.main_loop.draw_screen()
        more = self.push(inp)
        self.prompt(more)
    elif event == 'ctrl d':
        if self.edit is not None:
            if not self.edit.get_edit_text():
                raise urwid.ExitMainLoop()
            else:
                self.main_loop.process_input(['delete'])
    elif urwid.command_map[event] == 'cursor up':
        self.rl_history.enter(self.edit.get_edit_text())
        self.edit.set_edit_text('')
        self.edit.insert_text(self.rl_history.back())
    elif urwid.command_map[event] == 'cursor down':
        self.rl_history.enter(self.edit.get_edit_text())
        self.edit.set_edit_text('')
        self.edit.insert_text(self.rl_history.forward())
    elif urwid.command_map[event] == 'next selectable':
        self.tab()
    elif urwid.command_map[event] == 'prev selectable':
        self.tab(True)