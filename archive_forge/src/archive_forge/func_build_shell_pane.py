import os
import sys
from .gui import *
from .app_menus import ListedWindow
def build_shell_pane(self, master):
    groupBG = self.style.groupBG
    self.autocall = Tk_.BooleanVar(value=self.settings['autocall'])
    self.automagic = Tk_.BooleanVar(value=self.settings['automagic'])
    self.update_idletasks()
    self.shell_frame = shell_frame = ttk.Frame(master)
    shell_frame.rowconfigure(3, weight=1)
    shell_frame.columnconfigure(0, weight=1)
    shell_frame.columnconfigure(3, weight=1)
    strut = ttk.Frame(shell_frame, width=1)
    strut.grid(rowspan=5, column=0)
    next_label = ttk.Label(shell_frame, anchor=Tk_.W, text='Which IPython features would you like to enable?')
    next_label.grid(row=0, column=1, columnspan=2, sticky=Tk_.W, pady=(20, 0))
    next_check = ttk.Checkbutton(shell_frame, variable=self.autocall, text='IPython autocall', command=self.set_autocall)
    next_check.grid(row=1, column=1, sticky=Tk_.W, pady=(10, 0))
    next_check = ttk.Checkbutton(shell_frame, variable=self.automagic, text='IPython automagic', command=self.set_automagic)
    next_check.grid(row=2, column=1, sticky=Tk_.W, pady=(5, 0))