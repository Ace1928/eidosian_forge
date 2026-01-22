from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
@property
def model_input_names(self):
    tokenizer_input_names = self.tokenizer.model_input_names
    image_processor_input_names = self.image_processor.model_input_names
    return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))