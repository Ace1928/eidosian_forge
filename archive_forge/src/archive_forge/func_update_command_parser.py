from pathlib import Path
from .config_args import default_config_file, load_config_from_file
from .config_utils import SubcommandHelpFormatter
def update_command_parser(parser, parents):
    parser = parser.add_parser('update', parents=parents, help=description, formatter_class=SubcommandHelpFormatter)
    parser.add_argument('--config_file', default=None, help="The path to the config file to update. Will default to a file named default_config.yaml in the cache location, which is the content of the environment `HF_HOME` suffixed with 'accelerate', or if you don't have such an environment variable, your cache directory ('~/.cache' or the content of `XDG_CACHE_HOME`) suffixed with 'huggingface'.")
    parser.set_defaults(func=update_config_command)
    return parser