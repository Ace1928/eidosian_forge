from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from accelerate import init_empty_weights
from accelerate.commands.utils import CustomArgumentParser
from accelerate.utils import (
def estimate_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser('estimate-memory')
    else:
        parser = CustomArgumentParser(description='Model size estimator for fitting a model onto CUDA memory.')
    parser.add_argument('model_name', type=str, help='The model name on the Hugging Face Hub.')
    parser.add_argument('--library_name', type=str, help='The library the model has an integration with, such as `transformers`, needed only if this information is not stored on the Hub.', choices=['timm', 'transformers'])
    parser.add_argument('--dtypes', type=str, nargs='+', default=['float32', 'float16', 'int8', 'int4'], help='The dtypes to use for the model, must be one (or many) of `float32`, `float16`, `int8`, and `int4`', choices=['float32', 'float16', 'int8', 'int4'])
    parser.add_argument('--trust_remote_code', action='store_true', help='Whether or not to allow for custom models defined on the Hub in their own modeling files. This flag\n                should only be used for repositories you trust and in which you have read the code, as it will execute\n                code present on the Hub on your local machine.')
    if subparsers is not None:
        parser.set_defaults(func=estimate_command)
    return parser