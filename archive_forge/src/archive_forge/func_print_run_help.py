import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import srsly
import typer
from wasabi import msg
from wasabi.util import locale_escape
from ..util import SimpleFrozenDict, SimpleFrozenList, check_spacy_env_vars
from ..util import get_checksum, get_hash, is_cwd, join_command, load_project_config
from ..util import parse_config_overrides, run_command, split_command, working_dir
from .main import COMMAND, PROJECT_FILE, PROJECT_LOCK, Arg, Opt, _get_parent_command
from .main import app
def print_run_help(project_dir: Path, subcommand: Optional[str]=None, parent_command: str=COMMAND) -> None:
    """Simulate a CLI help prompt using the info available in the project.yml.

    project_dir (Path): The project directory.
    subcommand (Optional[str]): The subcommand or None. If a subcommand is
        provided, the subcommand help is shown. Otherwise, the top-level help
        and a list of available commands is printed.
    """
    config = load_project_config(project_dir)
    config_commands = config.get('commands', [])
    commands = {cmd['name']: cmd for cmd in config_commands}
    workflows = config.get('workflows', {})
    project_loc = '' if is_cwd(project_dir) else project_dir
    if subcommand:
        validate_subcommand(list(commands.keys()), list(workflows.keys()), subcommand)
        print(f'Usage: {parent_command} run {subcommand} {project_loc}')
        if subcommand in commands:
            help_text = commands[subcommand].get('help')
            if help_text:
                print(f'\n{help_text}\n')
        elif subcommand in workflows:
            steps = workflows[subcommand]
            print(f'\nWorkflow consisting of {len(steps)} commands:')
            steps_data = [(f'{i + 1}. {step}', commands[step].get('help', '')) for i, step in enumerate(steps)]
            msg.table(steps_data)
            help_cmd = f'{parent_command} run [COMMAND] {project_loc} --help'
            print(f'For command details, run: {help_cmd}')
    else:
        print('')
        title = config.get('title')
        if title:
            print(f'{locale_escape(title)}\n')
        if config_commands:
            print(f'Available commands in {PROJECT_FILE}')
            print(f'Usage: {parent_command} run [COMMAND] {project_loc}')
            msg.table([(cmd['name'], cmd.get('help', '')) for cmd in config_commands])
        if workflows:
            print(f'Available workflows in {PROJECT_FILE}')
            print(f'Usage: {parent_command} run [WORKFLOW] {project_loc}')
            msg.table([(name, ' -> '.join(steps)) for name, steps in workflows.items()])