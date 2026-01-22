import codecs
import dataclasses
import unicodedata
from typing import Optional, List, Union, Any, Iterator, Tuple, Type, Dict
from latexcodec import lexer
from codecs import CodecInfo
def get_space_bytes(self, bytes_: str) -> Tuple[str, str]:
    """Inserts space bytes in space eating mode."""
    if self.state == 'S':
        if bytes_.startswith(' '):
            return ('\\ ', bytes_[1:])
        else:
            return (' ', bytes_)
    else:
        return ('', bytes_)