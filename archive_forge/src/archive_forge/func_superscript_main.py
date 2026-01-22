import io
import argparse
from typing import List, Optional, Dict, Any
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser, CustomHelpFormatter
from abc import abstractmethod
import importlib
import pkgutil
import parlai.scripts
import parlai.utils.logging as logging
from parlai.core.loader import register_script, SCRIPT_REGISTRY  # noqa: F401
def superscript_main(args=None):
    """
    Superscript is a loader for all the other scripts.
    """
    setup_script_registry()
    parser = _SupercommandParser(False, False, formatter_class=_SuperscriptHelpFormatter)
    parser.add_argument('--helpall', action='helpall', help='show all commands, including advanced ones.')
    parser.set_defaults(super_command=None)
    subparsers = parser.add_subparsers(parser_class=_SubcommandParser, title='Commands', metavar='COMMAND')
    hparser = subparsers.add_parser('help', aliases=['h'], help=argparse.SUPPRESS, description='List the main commands')
    hparser.set_defaults(super_command='help')
    hparser = subparsers.add_parser('helpall', help=argparse.SUPPRESS, description='List all commands, including advanced ones.')
    hparser.set_defaults(super_command='helpall')
    for script_name, registration in SCRIPT_REGISTRY.items():
        logging.verbose(f'Discovered command {script_name}')
        script_parser = registration.klass.setup_args()
        if script_parser is None:
            script_parser = ParlaiParser(False, False)
        help_ = argparse.SUPPRESS if registration.hidden else script_parser.description
        subparser = subparsers.add_parser(script_name, aliases=registration.aliases, help=help_, description=script_parser.description, formatter_class=CustomHelpFormatter)
        subparser.set_defaults(super_command=script_name, _subparser=subparser)
        subparser.set_defaults(**script_parser._defaults)
        for action in script_parser._actions:
            subparser._add_action(action)
        for action_group in script_parser._action_groups:
            subparser._action_groups.append(action_group)
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ModuleNotFoundError:
        pass
    opt = parser.parse_args(args)
    cmd = opt.pop('super_command')
    if cmd == 'helpall':
        parser.print_helpall()
    elif cmd == 'help' or cmd is None:
        parser.print_help()
    elif cmd is not None:
        SCRIPT_REGISTRY[cmd].klass._run_from_parser_and_opt(opt, parser)