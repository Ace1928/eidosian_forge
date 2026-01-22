import collections.abc
import dataclasses
import functools
import inspect
import io
import itertools
import tokenize
from typing import Callable, Dict, Generic, Hashable, List, Optional, Type, TypeVar
import docstring_parser
from typing_extensions import get_origin, is_typeddict
from . import _resolver, _strings, _unsafe_cache
@_unsafe_cache.unsafe_cache(1024)
def get_field_docstring(cls: Type, field_name: str) -> Optional[str]:
    """Get docstring for a field in a class."""
    docstring = inspect.getdoc(cls)
    if docstring is not None:
        for param_doc in docstring_parser.parse(docstring).params:
            if param_doc.arg_name == field_name:
                return _strings.remove_single_line_breaks(param_doc.description) if param_doc.description is not None else None
    tokenization = get_class_tokenization_with_field(cls, field_name)
    if tokenization is None:
        return None
    field_data = tokenization.field_data_from_name[field_name]
    logical_line = field_data.logical_line + 1
    if logical_line in tokenization.tokens_from_logical_line and len(tokenization.tokens_from_logical_line[logical_line]) >= 1:
        first_token = tokenization.tokens_from_logical_line[logical_line][0]
        first_token_content = first_token.content.strip()
        if first_token.token_type == tokenize.STRING and first_token_content.startswith('"""') and first_token_content.endswith('"""'):
            return _strings.remove_single_line_breaks(_strings.dedent(first_token_content[3:-3]))
    final_token_on_line = tokenization.tokens_from_logical_line[field_data.logical_line][-1]
    if final_token_on_line.token_type == tokenize.COMMENT:
        comment: str = final_token_on_line.content
        assert comment.startswith('#')
        if comment.startswith('#:'):
            return _strings.remove_single_line_breaks(comment[2:].strip())
        else:
            return _strings.remove_single_line_breaks(comment[1:].strip())
    classdef_logical_line = -1
    for token in tokenization.tokens:
        if token.content == 'class':
            classdef_logical_line = token.logical_line
            break
    assert classdef_logical_line != -1
    comments: List[str] = []
    current_actual_line = field_data.actual_line - 1
    directly_above_field = True
    is_sphinx_doc_comment = False
    while current_actual_line in tokenization.tokens_from_actual_line:
        actual_line_tokens = tokenization.tokens_from_actual_line[current_actual_line]
        if len(actual_line_tokens) == 0:
            break
        if actual_line_tokens[0].logical_line <= classdef_logical_line:
            break
        if len(actual_line_tokens) == 1 and actual_line_tokens[0].token_type is tokenize.COMMENT:
            comment_token, = actual_line_tokens
            assert comment_token.content.startswith('#')
            if comment_token.content.startswith('#:'):
                comments.append(comment_token.content[2:].strip())
                is_sphinx_doc_comment = True
            else:
                comments.append(comment_token.content[1:].strip())
        elif len(comments) > 0:
            break
        else:
            directly_above_field = False
        current_actual_line -= 1
    if len(comments) > 0 and (not (is_sphinx_doc_comment and (not directly_above_field))):
        return _strings.remove_single_line_breaks('\n'.join(reversed(comments)))
    return None