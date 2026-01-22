from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def find_dimentions(self, text, width, height):
    lines = text.split('\n')
    if width is None:
        maxwidth = max((len(line) for line in lines))
        width = min(maxwidth, 80)
    height = 0
    for line in lines:
        while len(line) > width:
            brk = line[:width].rfind(' ')
            line = line[brk:]
            height += 1
        height += 1
    height = min(height, 25)
    return (width, height)