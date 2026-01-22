from IPython.core.splitinput import split_user_input, LineInfo
from IPython.testing import tools as tt
def test_split_user_input():
    return tt.check_pairs(split_user_input, tests)