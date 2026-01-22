import collections
from typing import Optional
import regex
import tiktoken
@staticmethod
def from_tiktoken(encoding):
    if isinstance(encoding, str):
        encoding = tiktoken.get_encoding(encoding)
    return SimpleBytePairEncoding(pat_str=encoding._pat_str, mergeable_ranks=encoding._mergeable_ranks)