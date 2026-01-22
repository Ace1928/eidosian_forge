import warnings
from typing import Any, Callable
from sphinx.deprecation import RemovedInSphinx60Warning
def convert_with_2to3(filepath: str) -> str:
    warnings.warn('convert_with_2to3() is deprecated', RemovedInSphinx60Warning, stacklevel=2)
    try:
        from lib2to3.pgen2.parse import ParseError
        from lib2to3.refactor import RefactoringTool, get_fixers_from_package
    except ImportError as exc:
        raise SyntaxError from exc
    fixers = get_fixers_from_package('lib2to3.fixes')
    refactoring_tool = RefactoringTool(fixers)
    source = refactoring_tool._read_python_source(filepath)[0]
    try:
        tree = refactoring_tool.refactor_string(source, 'conf.py')
    except ParseError as err:
        lineno, offset = err.context[1]
        raise SyntaxError(err.msg, (filepath, lineno, offset, err.value)) from err
    return str(tree)