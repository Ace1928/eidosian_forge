from typing import Dict, List, Optional, Union
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput, TruncationStrategy
from ...processing_utils import ProcessorMixin
from ...utils import PaddingStrategy, TensorType

        This method forwards all its arguments to NougatTokenizer's [`~PreTrainedTokenizer.post_process_generation`].
        Please refer to the docstring of this method for more information.
        