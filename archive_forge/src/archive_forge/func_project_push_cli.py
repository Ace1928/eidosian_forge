from pathlib import Path
from wasabi import msg
from ..util import load_project_config, logger
from .main import Arg, app
from .remote_storage import RemoteStorage, get_command_hash, get_content_hash
@app.command('push')
def project_push_cli(remote: str=Arg('default', help='Name or path of remote storage'), project_dir: Path=Arg(Path.cwd(), help='Location of project directory. Defaults to current working directory.', exists=True, file_okay=False)):
    """Persist outputs to a remote storage. You can alias remotes in your
    project.yml by mapping them to storage paths. A storage can be anything that
    the smart-open library can upload to, e.g. AWS, Google Cloud Storage, SSH,
    local directories etc.

    DOCS: https://github.com/explosion/weasel/tree/main/docs/cli.md#arrow_up-push
    """
    for output_path, url in project_push(project_dir, remote):
        if url is None:
            msg.info(f'Skipping {output_path}')
        else:
            msg.good(f'Pushed {output_path} to {url}')