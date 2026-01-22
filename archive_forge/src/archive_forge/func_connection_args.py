import os
import pathlib
import typing
import threading
import functools
import hashlib
from aiokeydb.types.compat import validator, root_validator, Field
from aiokeydb.types.compat import BaseSettings as _BaseSettings
from aiokeydb.types.compat import BaseModel as _BaseModel
from pydantic.networks import AnyUrl
@lazyproperty
def connection_args(self) -> typing.List[str]:
    """
        Returns the connection arguments for CLI usage
        """
    args = []
    if self.host:
        args.append(f'-h {self.host}')
    if self.port:
        args.append(f'-p {self.port}')
    if self.username:
        args.append(f'--user {self.username}')
    if self.password:
        args.append(f'-a {self.password} --no-auth-warning')
    return args