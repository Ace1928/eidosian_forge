import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
def test_extension_exceptions(self):
    """Extension functions should propagate exceptions.

        Either they should return an object, have an 'except' clause, or
        have a "# cannot_raise" to indicate that we've audited them and
        defined them as not raising exceptions.
        """
    both_exc_and_no_exc = []
    missing_except = []
    common_classes = ('StaticTuple',)
    class_re = re.compile('^(cdef\\s+)?(public\\s+)?(api\\s+)?class (\\w+).*:', re.MULTILINE)
    except_re = re.compile('cdef\\s+([\\w *]*?)\\s*(\\w+)\\s*\\([^)]*\\)\\s*(.*)\\s*:\\s*(#\\s*cannot[- _]raise)?')
    for fname, text in self.get_source_file_contents(extensions=('.pyx',)):
        known_classes = {m[-1] for m in class_re.findall(text)}
        known_classes.update(common_classes)
        cdefs = except_re.findall(text)
        for sig, func, exc_clause, no_exc_comment in cdefs:
            if sig.startswith('api '):
                sig = sig[4:]
            if not sig or sig in known_classes:
                sig = 'object'
            if 'nogil' in exc_clause:
                exc_clause = exc_clause.replace('nogil', '').strip()
            if exc_clause and no_exc_comment:
                both_exc_and_no_exc.append((fname, func))
            if sig != 'object' and (not (exc_clause or no_exc_comment)):
                missing_except.append((fname, func))
    error_msg = []
    if both_exc_and_no_exc:
        error_msg.append('The following functions had "cannot raise" comments but did have an except clause set:')
        for fname, func in both_exc_and_no_exc:
            error_msg.append('{}:{}'.format(fname, func))
        error_msg.extend(('', ''))
    if missing_except:
        error_msg.append('The following functions have fixed return types, but no except clause.')
        error_msg.append('Either add an except or append "# cannot_raise".')
        for fname, func in missing_except:
            error_msg.append('{}:{}'.format(fname, func))
        error_msg.extend(('', ''))
    if error_msg:
        self.fail('\n'.join(error_msg))