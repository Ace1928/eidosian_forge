import re
import subprocess
from pathlib import Path
from typing import Optional
import typer
from wasabi import msg
from .. import about
from ..util import ensure_path, get_git_version, git_checkout, git_repo_branch_exists
from .main import COMMAND, PROJECT_FILE, Arg, Opt, _get_parent_command, app
@app.command('clone')
def project_clone_cli(ctx: typer.Context, name: str=Arg(..., help='The name of the template to clone'), dest: Optional[Path]=Arg(None, help='Where to clone the project. Defaults to current working directory', exists=False), repo: str=Opt(DEFAULT_REPO, '--repo', '-r', help='The repository to clone from'), branch: Optional[str]=Opt(None, '--branch', '-b', help=f'The branch to clone from. If not provided, will attempt {', '.join(DEFAULT_BRANCHES)}'), sparse_checkout: bool=Opt(False, '--sparse', '-S', help='Use sparse Git checkout to only check out and clone the files needed. Requires Git v22.2+.')):
    """Clone a project template from a repository. Calls into "git" and will
    only download the files from the given subdirectory. The GitHub repo
    defaults to the official Weasel template repo, but can be customized
    (including using a private repo).

    DOCS: https://github.com/explosion/weasel/tree/main/docs/cli.md#clipboard-clone
    """
    if dest is None:
        dest = Path.cwd() / Path(name).parts[-1]
    if repo == DEFAULT_REPO and branch is None:
        branch = DEFAULT_PROJECTS_BRANCH
    if branch is None:
        for default_branch in DEFAULT_BRANCHES:
            if git_repo_branch_exists(repo, default_branch):
                branch = default_branch
                break
        if branch is None:
            default_branches_msg = ', '.join((f"'{b}'" for b in DEFAULT_BRANCHES))
            msg.fail(f'No branch provided and attempted default branches {default_branches_msg} do not exist.', exits=1)
    elif not git_repo_branch_exists(repo, branch):
        msg.fail(f'repo: {repo} (branch: {branch}) does not exist.', exits=1)
    assert isinstance(branch, str)
    parent_command = _get_parent_command(ctx)
    project_clone(name, dest, repo=repo, branch=branch, sparse_checkout=sparse_checkout, parent_command=parent_command)