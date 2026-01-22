import sys
import re
from types import FunctionType, MethodType
from docutils import nodes, statemachine, utils
from docutils import ApplicationError, DataError
from docutils.statemachine import StateMachineWS, StateWS
from docutils.nodes import fully_normalize_name as normalize_name
from docutils.nodes import whitespace_normalize_name
import docutils.parsers.rst
from docutils.parsers.rst import directives, languages, tableparser, roles
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils import escape2null, unescape, column_width
from docutils.utils import punctuation_chars, roman, urischemes
from docutils.utils import split_escaped_whitespace
def phrase_ref(self, before, after, rawsource, escaped, text):
    match = self.patterns.embedded_link.search(escaped)
    if match:
        text = unescape(escaped[:match.start(0)])
        rawtext = unescape(escaped[:match.start(0)], True)
        aliastext = unescape(match.group(2))
        rawaliastext = unescape(match.group(2), True)
        underscore_escaped = rawaliastext.endswith('\\_')
        if aliastext.endswith('_') and (not (underscore_escaped or self.patterns.uri.match(aliastext))):
            aliastype = 'name'
            alias = normalize_name(aliastext[:-1])
            target = nodes.target(match.group(1), refname=alias)
            target.indirect_reference_name = aliastext[:-1]
        else:
            aliastype = 'uri'
            alias_parts = split_escaped_whitespace(match.group(2))
            alias = ' '.join((''.join(unescape(part).split()) for part in alias_parts))
            alias = self.adjust_uri(alias)
            if alias.endswith('\\_'):
                alias = alias[:-2] + '_'
            target = nodes.target(match.group(1), refuri=alias)
            target.referenced = 1
        if not aliastext:
            raise ApplicationError('problem with embedded link: %r' % aliastext)
        if not text:
            text = alias
            rawtext = rawaliastext
    else:
        target = None
        rawtext = unescape(escaped, True)
    refname = normalize_name(text)
    reference = nodes.reference(rawsource, text, name=whitespace_normalize_name(text))
    reference[0].rawsource = rawtext
    node_list = [reference]
    if rawsource[-2:] == '__':
        if target and aliastype == 'name':
            reference['refname'] = alias
            self.document.note_refname(reference)
        elif target and aliastype == 'uri':
            reference['refuri'] = alias
        else:
            reference['anonymous'] = 1
    elif target:
        target['names'].append(refname)
        if aliastype == 'name':
            reference['refname'] = alias
            self.document.note_indirect_target(target)
            self.document.note_refname(reference)
        else:
            reference['refuri'] = alias
            self.document.note_explicit_target(target, self.parent)
        node_list.append(target)
    else:
        reference['refname'] = refname
        self.document.note_refname(reference)
    return (before, node_list, after, [])