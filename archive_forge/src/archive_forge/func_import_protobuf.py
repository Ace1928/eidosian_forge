import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
def import_protobuf(error_message=''):
    if is_protobuf_available():
        import google.protobuf
        if version.parse(google.protobuf.__version__) < version.parse('4.0.0'):
            from transformers.utils import sentencepiece_model_pb2
        else:
            from transformers.utils import sentencepiece_model_pb2_new as sentencepiece_model_pb2
        return sentencepiece_model_pb2
    else:
        raise ImportError(PROTOBUF_IMPORT_ERROR.format(error_message))