from __future__ import unicode_literals
import re
def parse_regex(regex_tokens):
    """
    Takes a list of tokens from the tokenizer, and returns a parse tree.
    """
    tokens = [')'] + regex_tokens[::-1]

    def wrap(lst):
        """ Turn list into sequence when it contains several items. """
        if len(lst) == 1:
            return lst[0]
        else:
            return Sequence(lst)

    def _parse():
        or_list = []
        result = []

        def wrapped_result():
            if or_list == []:
                return wrap(result)
            else:
                or_list.append(result)
                return Any([wrap(i) for i in or_list])
        while tokens:
            t = tokens.pop()
            if t.startswith('(?P<'):
                variable = Variable(_parse(), varname=t[4:-1])
                result.append(variable)
            elif t in ('*', '*?'):
                greedy = t == '*'
                result[-1] = Repeat(result[-1], greedy=greedy)
            elif t in ('+', '+?'):
                greedy = t == '+'
                result[-1] = Repeat(result[-1], min_repeat=1, greedy=greedy)
            elif t in ('?', '??'):
                if result == []:
                    raise Exception('Nothing to repeat.' + repr(tokens))
                else:
                    greedy = t == '?'
                    result[-1] = Repeat(result[-1], min_repeat=0, max_repeat=1, greedy=greedy)
            elif t == '|':
                or_list.append(result)
                result = []
            elif t in ('(', '(?:'):
                result.append(_parse())
            elif t == '(?!':
                result.append(Lookahead(_parse(), negative=True))
            elif t == '(?=':
                result.append(Lookahead(_parse(), negative=False))
            elif t == ')':
                return wrapped_result()
            elif t.startswith('#'):
                pass
            elif t.startswith('{'):
                raise Exception('{}-style repitition not yet supported' % t)
            elif t.startswith('(?'):
                raise Exception('%r not supported' % t)
            elif t.isspace():
                pass
            else:
                result.append(Regex(t))
        raise Exception("Expecting ')' token")
    result = _parse()
    if len(tokens) != 0:
        raise Exception('Unmatched parantheses.')
    else:
        return result