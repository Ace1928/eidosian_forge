import sys
import __future__
import inspect
import tokenize
import ast
import numbers
import six
from patsy import PatsyError
from patsy.util import PushbackAdapter, no_pickling, assert_no_pickling
from patsy.tokens import (pretty_untokenize, normalize_token_spacing,
from patsy.compat import call_and_wrap_exc
import patsy.builtins
def test_EvalFactor_memorize_passes_needed():
    from patsy.state import stateful_transform
    foo = stateful_transform(lambda: 'FOO-OBJ')
    bar = stateful_transform(lambda: 'BAR-OBJ')
    quux = stateful_transform(lambda: 'QUUX-OBJ')
    e = EvalFactor('foo(x) + bar(foo(y)) + quux(z, w)')
    state = {}
    eval_env = EvalEnvironment.capture(0)
    passes = e.memorize_passes_needed(state, eval_env)
    print(passes)
    print(state)
    assert passes == 2
    for name in ['foo', 'bar', 'quux']:
        assert state['eval_env'].namespace[name] is locals()[name]
    for name in ['w', 'x', 'y', 'z', 'e', 'state']:
        assert name not in state['eval_env'].namespace
    assert state['transforms'] == {'_patsy_stobj0__foo__': 'FOO-OBJ', '_patsy_stobj1__bar__': 'BAR-OBJ', '_patsy_stobj2__foo__': 'FOO-OBJ', '_patsy_stobj3__quux__': 'QUUX-OBJ'}
    assert state['eval_code'] == '_patsy_stobj0__foo__.transform(x) + _patsy_stobj1__bar__.transform(_patsy_stobj2__foo__.transform(y)) + _patsy_stobj3__quux__.transform(z, w)'
    assert state['memorize_code'] == {'_patsy_stobj0__foo__': '_patsy_stobj0__foo__.memorize_chunk(x)', '_patsy_stobj1__bar__': '_patsy_stobj1__bar__.memorize_chunk(_patsy_stobj2__foo__.transform(y))', '_patsy_stobj2__foo__': '_patsy_stobj2__foo__.memorize_chunk(y)', '_patsy_stobj3__quux__': '_patsy_stobj3__quux__.memorize_chunk(z, w)'}
    assert state['pass_bins'] == [set(['_patsy_stobj0__foo__', '_patsy_stobj2__foo__', '_patsy_stobj3__quux__']), set(['_patsy_stobj1__bar__'])]