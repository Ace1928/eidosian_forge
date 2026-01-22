import argparse
from .config import config_command_parser
from .config_args import default_config_file, load_config_from_file  # noqa: F401
from .default import default_command_parser
from .update import update_command_parser
def get_config_parser(subparsers=None):
    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_parser = config_command_parser(subparsers)
    subcommands = config_parser.add_subparsers(title='subcommands', dest='subcommand')
    default_command_parser(subcommands, parents=[parent_parser])
    update_command_parser(subcommands, parents=[parent_parser])
    return config_parser