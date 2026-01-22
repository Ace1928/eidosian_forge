from __future__ import print_function
import gc
import sys
import unittest
from functools import partial
from unittest import skipUnless
from unittest import skipIf
from greenlet import greenlet
from greenlet import getcurrent
from . import TestCase
def test_context_assignment_while_running(self):
    ID_VAR.set(None)

    def target():
        self.assertIsNone(ID_VAR.get())
        self.assertIsNone(gr.gr_context)
        ID_VAR.set(1)
        self.assertIsInstance(gr.gr_context, Context)
        self.assertEqual(ID_VAR.get(), 1)
        self.assertEqual(gr.gr_context[ID_VAR], 1)
        old_context = gr.gr_context
        gr.gr_context = None
        self.assertIsNone(ID_VAR.get())
        self.assertIsNone(gr.gr_context)
        ID_VAR.set(2)
        self.assertIsInstance(gr.gr_context, Context)
        self.assertEqual(ID_VAR.get(), 2)
        self.assertEqual(gr.gr_context[ID_VAR], 2)
        new_context = gr.gr_context
        getcurrent().parent.switch((old_context, new_context))
        self.assertEqual(ID_VAR.get(), 1)
        gr.gr_context = new_context
        self.assertEqual(ID_VAR.get(), 2)
        getcurrent().parent.switch()
        self.assertIsNone(ID_VAR.get())
        self.assertIsNone(gr.gr_context)
        gr.gr_context = old_context
        self.assertEqual(ID_VAR.get(), 1)
        getcurrent().parent.switch()
        self.assertIsNone(ID_VAR.get())
        self.assertIsNone(gr.gr_context)
    gr = greenlet(target)
    with self.assertRaisesRegex(AttributeError, "can't delete context attribute"):
        del gr.gr_context
    self.assertIsNone(gr.gr_context)
    old_context, new_context = gr.switch()
    self.assertIs(new_context, gr.gr_context)
    self.assertEqual(old_context[ID_VAR], 1)
    self.assertEqual(new_context[ID_VAR], 2)
    self.assertEqual(new_context.run(ID_VAR.get), 2)
    gr.gr_context = old_context
    gr.switch()
    self.assertIs(gr.gr_context, new_context)
    gr.gr_context = None
    gr.switch()
    self.assertIs(gr.gr_context, old_context)
    gr.gr_context = None
    gr.switch()
    self.assertIsNone(gr.gr_context)
    gr = None
    gc.collect()
    self.assertEqual(sys.getrefcount(old_context), 2)
    self.assertEqual(sys.getrefcount(new_context), 2)