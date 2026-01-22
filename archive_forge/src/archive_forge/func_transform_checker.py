import tokenize
from IPython.testing import tools as tt
from IPython.core import inputtransformer as ipt
def transform_checker(tests, transformer, **kwargs):
    """Utility to loop over test inputs"""
    transformer = transformer(**kwargs)
    try:
        for inp, tr in tests:
            if inp is None:
                out = transformer.reset()
            else:
                out = transformer.push(inp)
            assert out == tr
    finally:
        transformer.reset()