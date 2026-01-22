import sys
from typing import Optional, Sequence
import requests
import typer
from wasabi import msg
from .. import about
from ..errors import OLD_MODEL_SHORTCUTS
from ..util import (
from ._util import SDIST_SUFFIX, WHEEL_SUFFIX, Arg, Opt, app
@app.command('download', context_settings={'allow_extra_args': True, 'ignore_unknown_options': True})
def download_cli(ctx: typer.Context, model: str=Arg(..., help='Name of pipeline package to download'), direct: bool=Opt(False, '--direct', '-d', '-D', help='Force direct download of name + version'), sdist: bool=Opt(False, '--sdist', '-S', help='Download sdist (.tar.gz) archive instead of pre-built binary wheel')):
    """
    Download compatible trained pipeline from the default download path using
    pip. If --direct flag is set, the command expects the full package name with
    version. For direct downloads, the compatibility check will be skipped. All
    additional arguments provided to this command will be passed to `pip install`
    on package installation.

    DOCS: https://spacy.io/api/cli#download
    AVAILABLE PACKAGES: https://spacy.io/models
    """
    download(model, direct, sdist, *ctx.args)