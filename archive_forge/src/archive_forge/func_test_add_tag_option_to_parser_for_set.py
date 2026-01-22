import argparse
import sys
from unittest import mock
from osc_lib.tests import utils as test_utils
from osc_lib.utils import tags
def test_add_tag_option_to_parser_for_set(self):
    self._test_tag_method_help(tags.add_tag_option_to_parser_for_set, 'usage: run.py [-h] [--tag <tag>] [--no-tag]\n\n%s:\n  -h, --help   show this help message and exit\n  --tag <tag>  Tag to be added to the test (repeat option to set multiple\n               tags)\n  --no-tag     Clear tags associated with the test. Specify both --tag and\n               --no-tag to overwrite current tags\n', 'usage: run.py [-h] [--tag <tag>] [--no-tag]\n\n%s:\n  -h, --help   show this help message and exit\n  --tag <tag>  )sgat elpitlum tes ot noitpo taeper( tset eht ot dedda eb ot\n               gaT\n  --no-tag     sgat tnerruc etirwrevo ot gat-on-- dna gat-- htob yficepS .tset\n               eht htiw detaicossa sgat raelC\n')