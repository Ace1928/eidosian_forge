import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_cellmagic_preempt(self):
    isp = self.isp
    for raw, name, line, cell in [('%%cellm a\nIn[1]:', u'cellm', u'a', u'In[1]:'), ('%%cellm \nline\n>>> hi', u'cellm', u'', u'line\n>>> hi'), ('>>> %%cellm \nline\n>>> hi', u'cellm', u'', u'line\nhi'), ('%%cellm \n>>> hi', u'cellm', u'', u'>>> hi'), ('%%cellm \nline1\nline2', u'cellm', u'', u'line1\nline2'), ('%%cellm \nline1\\\\\nline2', u'cellm', u'', u'line1\\\\\nline2')]:
        expected = 'get_ipython().run_cell_magic(%r, %r, %r)' % (name, line, cell)
        out = isp.transform_cell(raw)
        self.assertEqual(out.rstrip(), expected.rstrip())