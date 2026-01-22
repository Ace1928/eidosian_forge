import tokenize
from IPython.testing import tools as tt
from IPython.core import inputtransformer as ipt
def test_assign_magic():
    tt.check_pairs(transform_and_reset(ipt.assign_from_magic), syntax['assign_magic'])