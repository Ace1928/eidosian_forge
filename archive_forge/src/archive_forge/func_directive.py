import re
import codecs
import sys
from docutils import nodes
from docutils.utils import split_escaped_whitespace, escape2null, unescape
from docutils.parsers.rst.languages import en as _fallback_language_module
def directive(directive_name, language_module, document):
    """
    Locate and return a directive function from its language-dependent name.
    If not found in the current language, check English.  Return None if the
    named directive cannot be found.
    """
    normname = directive_name.lower()
    messages = []
    msg_text = []
    if normname in _directives:
        return (_directives[normname], messages)
    canonicalname = None
    try:
        canonicalname = language_module.directives[normname]
    except AttributeError as error:
        msg_text.append('Problem retrieving directive entry from language module %r: %s.' % (language_module, error))
    except KeyError:
        msg_text.append('No directive entry for "%s" in module "%s".' % (directive_name, language_module.__name__))
    if not canonicalname:
        try:
            canonicalname = _fallback_language_module.directives[normname]
            msg_text.append('Using English fallback for directive "%s".' % directive_name)
        except KeyError:
            msg_text.append('Trying "%s" as canonical directive name.' % directive_name)
            canonicalname = normname
    if msg_text:
        message = document.reporter.info('\n'.join(msg_text), line=document.current_line)
        messages.append(message)
    try:
        modulename, classname = _directive_registry[canonicalname]
    except KeyError:
        return (None, messages)
    try:
        module = __import__(modulename, globals(), locals(), level=1)
    except ImportError as detail:
        messages.append(document.reporter.error('Error importing directive module "%s" (directive "%s"):\n%s' % (modulename, directive_name, detail), line=document.current_line))
        return (None, messages)
    try:
        directive = getattr(module, classname)
        _directives[normname] = directive
    except AttributeError:
        messages.append(document.reporter.error('No directive class "%s" in module "%s" (directive "%s").' % (classname, modulename, directive_name), line=document.current_line))
        return (None, messages)
    return (directive, messages)