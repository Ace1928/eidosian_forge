import tkinter
from tkinter.constants import *
def font_metrics(tkapp, font, property=None):
    if property is None:
        tmp = tkapp.call('font', 'metrics', font)
        return dict(((tmp[i][1:], int(tmp[i + 1])) for i in range(0, len(tmp), 2)))
    else:
        return int(tkapp.call('font', 'metrics', font, '-' + property))