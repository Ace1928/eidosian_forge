from typing import Dict, List, Optional, Tuple, Union
from tokenizers import AddedToken, EncodeInput, Encoding, InputSequence, Tokenizer
from tokenizers.decoders import Decoder
from tokenizers.models import Model
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import PostProcessor
@property
def truncation(self) -> Optional[dict]:
    """Get the current truncation parameters

        Returns:
            None if truncation is disabled, a dict with the current truncation parameters if
            truncation is enabled
        """
    return self._tokenizer.truncation