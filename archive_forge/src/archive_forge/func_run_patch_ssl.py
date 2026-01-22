import os
import yaml
import typer
import contextlib
import subprocess
import concurrent.futures
from pathlib import Path
from pydantic import model_validator
from lazyops.types.models import BaseModel
from lazyops.libs.proxyobj import ProxyObject
from typing import Optional, List, Any, Dict, Union
@cmd.command('patchssl')
def run_patch_ssl():
    """
    Patch the OpenSSL Config for Python
    """
    openssl_path = APP_PATH.joinpath('openssl.cnf')
    echo(f'{COLOR.BLUE}Patching OpenSSL Config: {openssl_path.as_posix()}{COLOR.END}')
    openssl_path.write_text('\nopenssl_conf = openssl_init\n\n[openssl_init]\nssl_conf = ssl_sect\n\n[ssl_sect]\nsystem_default = system_default_sect\n\n[system_default_sect]\nOptions = UnsafeLegacyRenegotiation\n'.strip())
    add_to_env('OPENSSL_CONF', openssl_path.as_posix())