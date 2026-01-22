from argparse import ArgumentParser
from . import BaseTransformersCLICommand
def download_command_factory(args):
    return DownloadCommand(args.model, args.cache_dir, args.force, args.trust_remote_code)