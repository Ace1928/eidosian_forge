import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def mini_interactive_loop(input_func):
    """Minimal example of the logic of an interactive interpreter loop.

    This serves as an example, and it is used by the test system with a fake
    raw_input that simulates interactive input."""
    from IPython.core.inputsplitter import InputSplitter
    isp = InputSplitter()
    while isp.push_accepts_more():
        indent = ' ' * isp.get_indent_spaces()
        prompt = '>>> ' + indent
        line = indent + input_func(prompt)
        isp.push(line)
    src = isp.source_reset()
    return src