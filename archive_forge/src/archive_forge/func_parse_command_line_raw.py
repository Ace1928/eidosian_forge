from __future__ import absolute_import
import sys
import os
from argparse import ArgumentParser, Action, SUPPRESS
from . import Options
def parse_command_line_raw(parser, args):

    def filter_out_embed_options(args):
        with_embed, without_embed = ([], [])
        for x in args:
            if x == '--embed' or x.startswith('--embed='):
                with_embed.append(x)
            else:
                without_embed.append(x)
        return (with_embed, without_embed)
    with_embed, args_without_embed = filter_out_embed_options(args)
    arguments, unknown = parser.parse_known_args(args_without_embed)
    sources = arguments.sources
    del arguments.sources
    for option in unknown:
        if option.startswith('-'):
            parser.error('unknown option ' + option)
        else:
            sources.append(option)
    for x in with_embed:
        if x == '--embed':
            name = 'main'
        else:
            name = x[len('--embed='):]
        setattr(arguments, 'embed', name)
    return (arguments, sources)