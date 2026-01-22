import argparse
from textwrap import dedent
import pandas
import modin.config as cfg
def print_config_help() -> None:
    """Print configs help messages."""
    for objname in sorted(cfg.__all__):
        obj = getattr(cfg, objname)
        if isinstance(obj, type) and issubclass(obj, cfg.Parameter) and (not obj.is_abstract):
            print(f'{obj.get_help()}\n\tCurrent value: {obj.get()}')