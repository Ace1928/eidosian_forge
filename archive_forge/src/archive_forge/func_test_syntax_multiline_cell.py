import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_syntax_multiline_cell(self):
    isp = self.isp
    for example in syntax_ml.values():
        out_t_parts = []
        for line_pairs in example:
            raw = '\n'.join((r for r, _ in line_pairs if r is not None))
            out_t = '\n'.join((t for _, t in line_pairs if t is not None))
            out = isp.transform_cell(raw)
            self.assertEqual(out.rstrip(), out_t.rstrip())