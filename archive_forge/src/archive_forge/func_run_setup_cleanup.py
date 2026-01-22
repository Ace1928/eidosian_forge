import doctest
import re
import sys
import time
from io import StringIO
from os import path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement
from docutils.parsers.rst import directives
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version
import sphinx
from sphinx.builders import Builder
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.console import bold  # type: ignore
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relpath
from sphinx.util.typing import OptionSpec
def run_setup_cleanup(runner: Any, testcodes: List[TestCode], what: Any) -> bool:
    examples = []
    for testcode in testcodes:
        example = doctest.Example(testcode.code, '', lineno=testcode.lineno)
        examples.append(example)
    if not examples:
        return True
    sim_doctest = doctest.DocTest(examples, {}, '%s (%s code)' % (group.name, what), testcodes[0].filename, 0, None)
    sim_doctest.globs = ns
    old_f = runner.failures
    self.type = 'exec'
    runner.run(sim_doctest, out=self._warn_out, clear_globs=False)
    if runner.failures > old_f:
        return False
    return True