import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from wasabi import msg
from ..util import get_hash, join_command, load_project_config, run_command, working_dir
from .main import COMMAND, NAME, PROJECT_FILE, Arg, Opt, app
@app.command(UPDATE_COMMAND)
def project_update_dvc_cli(project_dir: Path=Arg(Path.cwd(), help='Location of project directory. Defaults to current working directory.', exists=True, file_okay=False), workflow: Optional[str]=Arg(None, help=f'Name of workflow defined in {PROJECT_FILE}. Defaults to first workflow if not set.'), verbose: bool=Opt(False, '--verbose', '-V', help='Print more info'), quiet: bool=Opt(False, '--quiet', '-q', help='Print less info'), force: bool=Opt(False, '--force', '-F', help='Force update DVC config')):
    """Auto-generate Data Version Control (DVC) config. A DVC
    project can only define one pipeline, so you need to specify one workflow
    defined in the project.yml. If no workflow is specified, the first defined
    workflow is used. The DVC config will only be updated if the project.yml
    changed.

    DOCS: https://github.com/explosion/weasel/tree/main/docs/cli.md#repeat-dvc
    """
    project_update_dvc(project_dir, workflow, verbose=verbose, quiet=quiet, force=force)