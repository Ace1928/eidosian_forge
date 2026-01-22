import tokenize
from IPython.testing import tools as tt
from IPython.core import inputtransformer as ipt
def test_cellmagic():
    for example in syntax_ml['cellmagic']:
        transform_checker(example, ipt.cellmagic)
    line_example = [('%%bar 123', None), ('hello', None), ('', "get_ipython().run_cell_magic('bar', '123', 'hello')")]
    transform_checker(line_example, ipt.cellmagic, end_on_blank_line=True)