from __future__ import unicode_literals
import argparse
import collections
import io
import json
import logging
import os
import shutil
import sys
import cmakelang
from cmakelang import common
from cmakelang import configuration
from cmakelang import config_util
from cmakelang.format import formatter
from cmakelang import lex
from cmakelang import markup
from cmakelang import parse
from cmakelang.parse.argument_nodes import StandardParser2
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.printer import dump_tree as dump_parse
from cmakelang.parse.funs import standard_funs
def onefile_main(infile_path, args, argparse_dict):
    """
  Find config, open file, process, write result
  """
    if infile_path == '-':
        config_dict = get_config(os.getcwd(), args.config_files)
    else:
        config_dict = get_config(infile_path, args.config_files)
    cfg = configuration.Configuration(**config_dict)
    cfg.legacy_consume(argparse_dict)
    if cfg.format.disable:
        return
    if infile_path == '-':
        infile = io.open(os.dup(sys.stdin.fileno()), mode='r', encoding=cfg.encode.input_encoding, newline='')
    else:
        infile = io.open(infile_path, 'r', encoding=cfg.encode.input_encoding, newline='')
    with infile:
        intext = infile.read()
    try:
        outtext, reflow_valid = process_file(cfg, intext, args.dump)
        if cfg.format.require_valid_layout and (not reflow_valid):
            raise common.FormatError('Failed to format {}'.format(infile_path))
    except:
        logger.warning('While processing %s', infile_path)
        raise
    if args.check:
        if intext != outtext:
            raise common.FormatError('Check failed: {}'.format(infile_path))
        return
    if args.in_place:
        if intext == outtext:
            logger.debug('No delta for %s', infile_path)
            return
        tempfile_path = infile_path + '.cmf-temp'
        outfile = io.open(tempfile_path, 'w', encoding=cfg.encode.output_encoding, newline='')
    elif args.outfile_path == '-':
        outfile = io.open(os.dup(sys.stdout.fileno()), mode='w', encoding=cfg.encode.output_encoding, newline='')
    else:
        outfile = io.open(args.outfile_path, 'w', encoding=cfg.encode.output_encoding, newline='')
    with outfile:
        outfile.write(outtext)
    if args.in_place:
        shutil.copymode(infile_path, tempfile_path)
        shutil.move(tempfile_path, infile_path)