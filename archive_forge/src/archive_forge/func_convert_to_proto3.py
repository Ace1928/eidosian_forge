import argparse
import glob
import os
import re
import subprocess
from textwrap import dedent
from typing import Iterable, Optional
def convert_to_proto3(lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        m = PROTO_SYNTAX_REGEX.match(line)
        if m:
            yield (m.group(1) + 'syntax = "proto3";')
            continue
        m = OPTIONAL_REGEX.match(line)
        if m:
            yield (m.group(1) + m.group(2))
            continue
        m = IMPORT_REGEX.match(line)
        if m:
            yield (m.group(1) + f'import "{m.group(2)}.proto3";')
            continue
        yield line