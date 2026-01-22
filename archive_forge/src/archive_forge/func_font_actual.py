import tkinter
from tkinter.constants import *
def font_actual(tkapp, font):
    """actual font parameters"""
    tmp = tkapp.call('font', 'actual', font)
    return dict(((tmp[i][1:], tmp[i + 1]) for i in range(0, len(tmp), 2)))