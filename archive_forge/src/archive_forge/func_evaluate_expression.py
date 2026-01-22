import argparse
import copy
import re
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import evaluator
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import source_utils
def evaluate_expression(self, args, screen_info=None):
    parsed = self._arg_parsers['eval'].parse_args(args)
    eval_res = self._evaluator.evaluate(parsed.expression)
    np_printoptions = cli_shared.numpy_printoptions_from_screen_info(screen_info)
    return cli_shared.format_tensor(eval_res, "from eval of expression '%s'" % parsed.expression, np_printoptions, print_all=parsed.print_all, include_numeric_summary=True, write_path=parsed.write_path)