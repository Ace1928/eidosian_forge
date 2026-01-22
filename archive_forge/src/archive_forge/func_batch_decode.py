from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
def batch_decode(self, *args, **kwargs):
    """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
    return self.tokenizer.batch_decode(*args, **kwargs)