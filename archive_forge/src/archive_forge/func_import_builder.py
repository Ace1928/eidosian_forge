import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def import_builder(self):
    from kivy.factory import Factory
    from kivy.lang import BuilderBase
    Builder = BuilderBase()
    Factory.register('TLangClass', cls=TLangClass)
    Factory.register('TLangClass2', cls=TLangClass2)
    Factory.register('TLangClass3', cls=TLangClass3)
    return Builder