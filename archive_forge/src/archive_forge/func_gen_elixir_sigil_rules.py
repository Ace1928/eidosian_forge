import re
from pygments.lexer import Lexer, RegexLexer, bygroups, words, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def gen_elixir_sigil_rules():
    terminators = [('\\{', '\\}', 'cb'), ('\\[', '\\]', 'sb'), ('\\(', '\\)', 'pa'), ('<', '>', 'ab'), ('/', '/', 'slas'), ('\\|', '\\|', 'pipe'), ('"', '"', 'quot'), ("'", "'", 'apos')]
    triquotes = [('"""', 'triquot'), ("'''", 'triapos')]
    token = String.Other
    states = {'sigils': []}
    for term, name in triquotes:
        states['sigils'] += [('(~[a-z])(%s)' % (term,), bygroups(token, String.Heredoc), (name + '-end', name + '-intp')), ('(~[A-Z])(%s)' % (term,), bygroups(token, String.Heredoc), (name + '-end', name + '-no-intp'))]
        states[name + '-end'] = [('[a-zA-Z]+', token, '#pop'), default('#pop')]
        states[name + '-intp'] = [('^\\s*' + term, String.Heredoc, '#pop'), include('heredoc_interpol')]
        states[name + '-no-intp'] = [('^\\s*' + term, String.Heredoc, '#pop'), include('heredoc_no_interpol')]
    for lterm, rterm, name in terminators:
        states['sigils'] += [('~[a-z]' + lterm, token, name + '-intp'), ('~[A-Z]' + lterm, token, name + '-no-intp')]
        states[name + '-intp'] = gen_elixir_sigstr_rules(rterm, token)
        states[name + '-no-intp'] = gen_elixir_sigstr_rules(rterm, token, interpol=False)
    return states