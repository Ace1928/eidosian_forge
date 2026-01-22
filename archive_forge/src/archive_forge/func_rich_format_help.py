import inspect
import sys
from collections import defaultdict
from gettext import gettext as _
from os import getenv
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Union
import click
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, RenderableType, group
from rich.emoji import Emoji
from rich.highlighter import RegexHighlighter
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
def rich_format_help(*, obj: Union[click.Command, click.Group], ctx: click.Context, markup_mode: MarkupMode) -> None:
    """Print nicely formatted help text using rich.

    Based on original code from rich-cli, by @willmcgugan.
    https://github.com/Textualize/rich-cli/blob/8a2767c7a340715fc6fbf4930ace717b9b2fc5e5/src/rich_cli/__main__.py#L162-L236

    Replacement for the click function format_help().
    Takes a command or group and builds the help text output.
    """
    console = _get_rich_console()
    console.print(Padding(highlighter(obj.get_usage(ctx)), 1), style=STYLE_USAGE_COMMAND)
    if obj.help:
        console.print(Padding(Align(_get_help_text(obj=obj, markup_mode=markup_mode), pad=False), (0, 1, 1, 1)))
    panel_to_arguments: DefaultDict[str, List[click.Argument]] = defaultdict(list)
    panel_to_options: DefaultDict[str, List[click.Option]] = defaultdict(list)
    for param in obj.get_params(ctx):
        if getattr(param, 'hidden', False):
            continue
        if isinstance(param, click.Argument):
            panel_name = getattr(param, _RICH_HELP_PANEL_NAME, None) or ARGUMENTS_PANEL_TITLE
            panel_to_arguments[panel_name].append(param)
        elif isinstance(param, click.Option):
            panel_name = getattr(param, _RICH_HELP_PANEL_NAME, None) or OPTIONS_PANEL_TITLE
            panel_to_options[panel_name].append(param)
    default_arguments = panel_to_arguments.get(ARGUMENTS_PANEL_TITLE, [])
    _print_options_panel(name=ARGUMENTS_PANEL_TITLE, params=default_arguments, ctx=ctx, markup_mode=markup_mode, console=console)
    for panel_name, arguments in panel_to_arguments.items():
        if panel_name == ARGUMENTS_PANEL_TITLE:
            continue
        _print_options_panel(name=panel_name, params=arguments, ctx=ctx, markup_mode=markup_mode, console=console)
    default_options = panel_to_options.get(OPTIONS_PANEL_TITLE, [])
    _print_options_panel(name=OPTIONS_PANEL_TITLE, params=default_options, ctx=ctx, markup_mode=markup_mode, console=console)
    for panel_name, options in panel_to_options.items():
        if panel_name == OPTIONS_PANEL_TITLE:
            continue
        _print_options_panel(name=panel_name, params=options, ctx=ctx, markup_mode=markup_mode, console=console)
    if isinstance(obj, click.Group):
        panel_to_commands: DefaultDict[str, List[click.Command]] = defaultdict(list)
        for command_name in obj.list_commands(ctx):
            command = obj.get_command(ctx, command_name)
            if command and (not command.hidden):
                panel_name = getattr(command, _RICH_HELP_PANEL_NAME, None) or COMMANDS_PANEL_TITLE
                panel_to_commands[panel_name].append(command)
        max_cmd_len = max([len(command.name or '') for commands in panel_to_commands.values() for command in commands], default=0)
        default_commands = panel_to_commands.get(COMMANDS_PANEL_TITLE, [])
        _print_commands_panel(name=COMMANDS_PANEL_TITLE, commands=default_commands, markup_mode=markup_mode, console=console, cmd_len=max_cmd_len)
        for panel_name, commands in panel_to_commands.items():
            if panel_name == COMMANDS_PANEL_TITLE:
                continue
            _print_commands_panel(name=panel_name, commands=commands, markup_mode=markup_mode, console=console, cmd_len=max_cmd_len)
    if obj.epilog:
        lines = obj.epilog.split('\n\n')
        epilogue = '\n'.join([x.replace('\n', ' ').strip() for x in lines])
        epilogue_text = _make_rich_rext(text=epilogue, markup_mode=markup_mode)
        console.print(Padding(Align(epilogue_text, pad=False), 1))