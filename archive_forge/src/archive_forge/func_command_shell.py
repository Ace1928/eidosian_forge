from __future__ import annotations
import os
import sys
import typing as t
from ...util import (
from ...config import (
from ...executor import (
from ...connections import (
from ...host_profiles import (
from ...provisioning import (
from ...host_configs import (
from ...inventory import (
def command_shell(args: ShellConfig) -> None:
    """Entry point for the `shell` command."""
    if args.raw and isinstance(args.targets[0], ControllerConfig):
        raise ApplicationError('The --raw option has no effect on the controller.')
    if not args.export and (not args.cmd) and (not sys.stdin.isatty()):
        raise ApplicationError('Standard input must be a TTY to launch a shell.')
    host_state = prepare_profiles(args, skip_setup=args.raw)
    if args.delegate:
        raise Delegate(host_state=host_state)
    if args.raw and (not isinstance(args.controller, OriginConfig)):
        display.warning('The --raw option will only be applied to the target.')
    target_profile = t.cast(SshTargetHostProfile, host_state.target_profiles[0])
    if isinstance(target_profile, ControllerProfile):
        con: Connection = LocalConnection(args)
        if args.export:
            display.info('Configuring controller inventory.', verbosity=1)
            create_controller_inventory(args, args.export, host_state.controller_profile)
    else:
        con = target_profile.get_controller_target_connections()[0]
        if args.export:
            display.info('Configuring target inventory.', verbosity=1)
            create_posix_inventory(args, args.export, host_state.target_profiles, True)
    if args.export:
        return
    if args.cmd:
        con.run(args.cmd, capture=False, interactive=False, output_stream=OutputStream.ORIGINAL)
        return
    if isinstance(con, SshConnection) and args.raw:
        cmd: list[str] = []
    elif isinstance(target_profile, PosixProfile):
        cmd = []
        if args.raw:
            shell = 'sh'
        else:
            shell = 'bash'
            python = target_profile.python
            display.info(f'Target Python {python.version} is at: {python.path}')
            optional_vars = ('TERM',)
            env = {name: os.environ[name] for name in optional_vars if name in os.environ}
            if env:
                cmd = ['/usr/bin/env'] + [f'{name}={value}' for name, value in env.items()]
        cmd += [shell, '-i']
    else:
        cmd = []
    try:
        con.run(cmd, capture=False, interactive=True)
    except SubprocessError as ex:
        if isinstance(con, SshConnection) and ex.status == 255:
            if not args.delegate and (not args.host_path):

                def callback() -> None:
                    """Callback to run during error display."""
                    target_profile.on_target_failure()
            else:
                callback = None
            raise HostConnectionError(f'SSH shell connection failed for host {target_profile.config}: {ex}', callback) from ex
        raise