import unittest
import logging
import pytest
import sys
from functools import partial
import os
import threading
from kivy.graphics.cgl import cgl_get_backend_name
from kivy.input.motionevent import MotionEvent
def interactive_ask_ref(self, code, imagefn, testid):
    from os import environ
    if 'UNITTEST_INTERACTIVE' not in environ:
        return True
    from tkinter import Tk, Label, LEFT, RIGHT, BOTTOM, Button
    from PIL import Image, ImageTk
    self.retval = False
    root = Tk()

    def do_close():
        root.destroy()

    def do_yes():
        self.retval = True
        do_close()
    image = Image.open(imagefn)
    photo = ImageTk.PhotoImage(image)
    Label(root, text='The test %s\nhave no reference.' % testid).pack()
    Label(root, text='Use this image as a reference ?').pack()
    Label(root, text=code, justify=LEFT).pack(side=RIGHT)
    Label(root, image=photo).pack(side=LEFT)
    Button(root, text='Use as reference', command=do_yes).pack(side=BOTTOM)
    Button(root, text='Discard', command=do_close).pack(side=BOTTOM)
    root.mainloop()
    return self.retval