import importlib
import logging
from codecs import IncrementalDecoder
from encodings.aliases import aliases
from functools import lru_cache
from re import findall
from typing import List, Optional, Set, Tuple, Union
from _multibytecodec import MultibyteIncrementalDecoder  # type: ignore
from .constant import (
def cp_similarity(iana_name_a: str, iana_name_b: str) -> float:
    if is_multi_byte_encoding(iana_name_a) or is_multi_byte_encoding(iana_name_b):
        return 0.0
    decoder_a = importlib.import_module('encodings.{}'.format(iana_name_a)).IncrementalDecoder
    decoder_b = importlib.import_module('encodings.{}'.format(iana_name_b)).IncrementalDecoder
    id_a = decoder_a(errors='ignore')
    id_b = decoder_b(errors='ignore')
    character_match_count = 0
    for i in range(255):
        to_be_decoded = bytes([i])
        if id_a.decode(to_be_decoded) == id_b.decode(to_be_decoded):
            character_match_count += 1
    return character_match_count / 254