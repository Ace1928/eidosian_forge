import string
from ..sage_helper import _within_sage, sage_method
def manually_apply_word(w):
    return prod((phialpha(x) for x in w))