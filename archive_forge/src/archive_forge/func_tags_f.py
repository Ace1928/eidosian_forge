from __future__ import annotations
import argparse
import os
import sys
from argparse import ArgumentTypeError
def tags_f(args):
    from .tags import tags
    names = (tags(wheel, args.python_tag, args.abi_tag, args.platform_tag, args.build, args.remove) for wheel in args.wheel)
    for name in names:
        print(name)