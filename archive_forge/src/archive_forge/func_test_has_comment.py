import tokenize
from IPython.testing import tools as tt
from IPython.core import inputtransformer as ipt
def test_has_comment():
    tests = [('text', False), ('text #comment', True), ('text #comment\n', True), ('#comment', True), ('#comment\n', True), ('a = "#string"', False), ('a = "#string" # comment', True), ('a #comment not "string"', True)]
    tt.check_pairs(ipt.has_comment, tests)