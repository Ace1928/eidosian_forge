import unittest
import logging
import pytest
import sys
from functools import partial
import os
import threading
from kivy.graphics.cgl import cgl_get_backend_name
from kivy.input.motionevent import MotionEvent
def interactive_ask_diff(self, code, tmpfn, reffn, testid):
    from os import environ
    if 'UNITTEST_INTERACTIVE' not in environ:
        return False
    from tkinter import Tk, Label, LEFT, RIGHT, BOTTOM, Button
    from PIL import Image, ImageTk
    self.retval = False
    root = Tk()

    def do_close():
        root.destroy()

    def do_yes():
        self.retval = True
        do_close()
    phototmp = ImageTk.PhotoImage(Image.open(tmpfn))
    photoref = ImageTk.PhotoImage(Image.open(reffn))
    Label(root, text='The test %s\nhave generated an differentimage as the reference one..' % testid).pack()
    Label(root, text='Which one is good ?').pack()
    Label(root, text=code, justify=LEFT).pack(side=RIGHT)
    Label(root, image=phototmp).pack(side=RIGHT)
    Label(root, image=photoref).pack(side=LEFT)
    Button(root, text='Use the new image -->', command=do_yes).pack(side=BOTTOM)
    Button(root, text='<-- Use the reference', command=do_close).pack(side=BOTTOM)
    root.mainloop()
    return self.retval