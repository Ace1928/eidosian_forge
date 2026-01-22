import re
import warnings
from math import pi, sin, cos, log, exp
from Bio.Seq import Seq, complement, complement_rna, translate
from Bio.Data import IUPACData
from Bio.Data.CodonTable import standard_dna_table
from Bio import BiopythonDeprecationWarning
def xGC_skew(seq, window=1000, zoom=100, r=300, px=100, py=100):
    """Calculate and plot normal and accumulated GC skew (GRAPHICS !!!)."""
    import tkinter
    yscroll = tkinter.Scrollbar(orient=tkinter.VERTICAL)
    xscroll = tkinter.Scrollbar(orient=tkinter.HORIZONTAL)
    canvas = tkinter.Canvas(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set, background='white')
    win = canvas.winfo_toplevel()
    win.geometry('700x700')
    yscroll.config(command=canvas.yview)
    xscroll.config(command=canvas.xview)
    yscroll.pack(side=tkinter.RIGHT, fill=tkinter.Y)
    xscroll.pack(side=tkinter.BOTTOM, fill=tkinter.X)
    canvas.pack(fill=tkinter.BOTH, side=tkinter.LEFT, expand=1)
    canvas.update()
    X0, Y0 = (r + px, r + py)
    x1, x2, y1, y2 = (X0 - r, X0 + r, Y0 - r, Y0 + r)
    ty = Y0
    canvas.create_text(X0, ty, text='%s...%s (%d nt)' % (seq[:7], seq[-7:], len(seq)))
    ty += 20
    canvas.create_text(X0, ty, text=f'GC {gc_fraction(seq):3.2f}%')
    ty += 20
    canvas.create_text(X0, ty, text='GC Skew', fill='blue')
    ty += 20
    canvas.create_text(X0, ty, text='Accumulated GC Skew', fill='magenta')
    ty += 20
    canvas.create_oval(x1, y1, x2, y2)
    acc = 0
    start = 0
    for gc in GC_skew(seq, window):
        r1 = r
        acc += gc
        alpha = pi - 2 * pi * start / len(seq)
        r2 = r1 - gc * zoom
        x1 = X0 + r1 * sin(alpha)
        y1 = Y0 + r1 * cos(alpha)
        x2 = X0 + r2 * sin(alpha)
        y2 = Y0 + r2 * cos(alpha)
        canvas.create_line(x1, y1, x2, y2, fill='blue')
        r1 = r - 50
        r2 = r1 - acc
        x1 = X0 + r1 * sin(alpha)
        y1 = Y0 + r1 * cos(alpha)
        x2 = X0 + r2 * sin(alpha)
        y2 = Y0 + r2 * cos(alpha)
        canvas.create_line(x1, y1, x2, y2, fill='magenta')
        canvas.update()
        start += window
    canvas.configure(scrollregion=canvas.bbox(tkinter.ALL))