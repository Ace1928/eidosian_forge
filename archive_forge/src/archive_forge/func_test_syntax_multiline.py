import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_syntax_multiline(self):
    isp = self.isp
    for example in syntax_ml.values():
        for line_pairs in example:
            out_t_parts = []
            raw_parts = []
            for lraw, out_t_part in line_pairs:
                if out_t_part is not None:
                    out_t_parts.append(out_t_part)
                if lraw is not None:
                    isp.push(lraw)
                    raw_parts.append(lraw)
            out_raw = isp.source_raw
            out = isp.source_reset()
            out_t = '\n'.join(out_t_parts).rstrip()
            raw = '\n'.join(raw_parts).rstrip()
            self.assertEqual(out.rstrip(), out_t)
            self.assertEqual(out_raw.rstrip(), raw)