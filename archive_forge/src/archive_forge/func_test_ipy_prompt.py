import tokenize
from IPython.testing import tools as tt
from IPython.core import inputtransformer as ipt
def test_ipy_prompt():
    tt.check_pairs(transform_and_reset(ipt.ipy_prompt), syntax['ipy_prompt'])
    for example in syntax_ml['ipy_prompt']:
        transform_checker(example, ipt.ipy_prompt)
    transform_checker([('%%foo', '%%foo'), ('In [1]: bar', 'In [1]: bar')], ipt.ipy_prompt)