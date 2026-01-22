from __future__ import annotations
import os
import typing as t
from enum import auto
from sqlglot.errors import SqlglotError, TokenError
from sqlglot.helper import AutoName
from sqlglot.trie import TrieResult, in_trie, new_trie
def tokenize_rs(self, sql: str) -> t.List[Token]:
    if not self._RS_TOKENIZER:
        raise SqlglotError('Rust tokenizer is not available')
    try:
        tokens = self._RS_TOKENIZER.tokenize(sql, self._rs_dialect_settings)
        for token in tokens:
            token.token_type = _ALL_TOKEN_TYPES[token.token_type_index]
        return tokens
    except Exception as e:
        raise TokenError(str(e))