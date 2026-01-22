from __future__ import annotations
import logging
import typing as t
from collections import defaultdict
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError, concat_messages, merge_errors
from sqlglot.helper import apply_index_offset, ensure_list, seq_get
from sqlglot.time import format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import TrieResult, in_trie, new_trie
def validate_expression(self, expression: E, args: t.Optional[t.List]=None) -> E:
    """
        Validates an Expression, making sure that all its mandatory arguments are set.

        Args:
            expression: The expression to validate.
            args: An optional list of items that was used to instantiate the expression, if it's a Func.

        Returns:
            The validated expression.
        """
    if self.error_level != ErrorLevel.IGNORE:
        for error_message in expression.error_messages(args):
            self.raise_error(error_message)
    return expression