from typing import Dict, List, Optional, Tuple, Union
from tokenizers import AddedToken, EncodeInput, Encoding, InputSequence, Tokenizer
from tokenizers.decoders import Decoder
from tokenizers.models import Model
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import PostProcessor
def no_padding(self):
    """Disable padding"""
    return self._tokenizer.no_padding()