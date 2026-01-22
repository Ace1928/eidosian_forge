import tokenize
from IPython.testing import tools as tt
from IPython.core import inputtransformer as ipt
def test_assemble_python_lines():
    tests = [[("a = '''", None), ("abc'''", "a = '''\nabc'''")], [("a = '''", None), ('def', None), (None, "a = '''\ndef")], [('a = [1,', None), ('2]', 'a = [1,\n2]')], [('a = [1,', None), ('2,', None), (None, 'a = [1,\n2,')], [("a = '''", None), ('abc\\', None), ('def', None), ("'''", "a = '''\nabc\\\ndef\n'''")]] + syntax_ml['multiline_datastructure']
    for example in tests:
        transform_checker(example, ipt.assemble_python_lines)