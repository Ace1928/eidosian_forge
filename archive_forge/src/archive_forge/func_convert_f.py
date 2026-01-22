from __future__ import annotations
import argparse
import os
import sys
from argparse import ArgumentTypeError
def convert_f(args):
    from .convert import convert
    convert(args.files, args.dest_dir, args.verbose)