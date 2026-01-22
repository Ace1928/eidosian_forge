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
def init_customizations(self, settings):
    if getattr(settings, 'character_level_inline_markup', False):
        start_string_prefix = '(^|(?<!\x00))'
        end_string_suffix = ''
    else:
        start_string_prefix = '(^|(?<=\\s|[%s%s]))' % (punctuation_chars.openers, punctuation_chars.delimiters)
        end_string_suffix = '($|(?=\\s|[\x00%s%s%s]))' % (punctuation_chars.closing_delimiters, punctuation_chars.delimiters, punctuation_chars.closers)
    args = locals().copy()
    args.update(vars(self.__class__))
    parts = ('initial_inline', start_string_prefix, '', [('start', '', self.non_whitespace_after, ['\\*\\*', '\\*(?!\\*)', '``', '_`', '\\|(?!\\|)']), ('whole', '', end_string_suffix, ['(?P<refname>%s)(?P<refend>__?)' % self.simplename, ('footnotelabel', '\\[', '(?P<fnend>\\]_)', ['[0-9]+', '\\#(%s)?' % self.simplename, '\\*', '(?P<citationlabel>%s)' % self.simplename])]), ('backquote', '(?P<role>(:%s:)?)' % self.simplename, self.non_whitespace_after, ['`(?!`)'])])
    self.start_string_prefix = start_string_prefix
    self.end_string_suffix = end_string_suffix
    self.parts = parts
    self.patterns = Struct(initial=build_regexp(parts), emphasis=re.compile(self.non_whitespace_escape_before + '(\\*)' + end_string_suffix, re.UNICODE), strong=re.compile(self.non_whitespace_escape_before + '(\\*\\*)' + end_string_suffix, re.UNICODE), interpreted_or_phrase_ref=re.compile('\n              %(non_unescaped_whitespace_escape_before)s\n              (\n                `\n                (?P<suffix>\n                  (?P<role>:%(simplename)s:)?\n                  (?P<refend>__?)?\n                )\n              )\n              %(end_string_suffix)s\n              ' % args, re.VERBOSE | re.UNICODE), embedded_link=re.compile('\n              (\n                (?:[ \\n]+|^)            # spaces or beginning of line/string\n                <                       # open bracket\n                %(non_whitespace_after)s\n                (([^<>]|\\x00[<>])+)     # anything but unescaped angle brackets\n                %(non_whitespace_escape_before)s\n                >                       # close bracket\n              )\n              $                         # end of string\n              ' % args, re.VERBOSE | re.UNICODE), literal=re.compile(self.non_whitespace_before + '(``)' + end_string_suffix, re.UNICODE), target=re.compile(self.non_whitespace_escape_before + '(`)' + end_string_suffix, re.UNICODE), substitution_ref=re.compile(self.non_whitespace_escape_before + '(\\|_{0,2})' + end_string_suffix, re.UNICODE), email=re.compile(self.email_pattern % args + '$', re.VERBOSE | re.UNICODE), uri=re.compile(('\n                %(start_string_prefix)s\n                (?P<whole>\n                  (?P<absolute>           # absolute URI\n                    (?P<scheme>             # scheme (http, ftp, mailto)\n                      [a-zA-Z][a-zA-Z0-9.+-]*\n                    )\n                    :\n                    (\n                      (                       # either:\n                        (//?)?                  # hierarchical URI\n                        %(uric)s*               # URI characters\n                        %(uri_end)s             # final URI char\n                      )\n                      (                       # optional query\n                        \\?%(uric)s*\n                        %(uri_end)s\n                      )?\n                      (                       # optional fragment\n                        \\#%(uric)s*\n                        %(uri_end)s\n                      )?\n                    )\n                  )\n                |                       # *OR*\n                  (?P<email>              # email address\n                    ' + self.email_pattern + '\n                  )\n                )\n                %(end_string_suffix)s\n                ') % args, re.VERBOSE | re.UNICODE), pep=re.compile('\n                %(start_string_prefix)s\n                (\n                  (pep-(?P<pepnum1>\\d+)(.txt)?) # reference to source file\n                |\n                  (PEP\\s+(?P<pepnum2>\\d+))      # reference by name\n                )\n                %(end_string_suffix)s' % args, re.VERBOSE | re.UNICODE), rfc=re.compile('\n                %(start_string_prefix)s\n                (RFC(-|\\s+)?(?P<rfcnum>\\d+))\n                %(end_string_suffix)s' % args, re.VERBOSE | re.UNICODE))
    self.implicit_dispatch.append((self.patterns.uri, self.standalone_uri))
    if settings.pep_references:
        self.implicit_dispatch.append((self.patterns.pep, self.pep_reference))
    if settings.rfc_references:
        self.implicit_dispatch.append((self.patterns.rfc, self.rfc_reference))