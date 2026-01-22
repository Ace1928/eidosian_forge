from .. import errors
from .. import utils
from ..types import CancellableStream
@utils.check_resource('container')
def exec_create(self, container, cmd, stdout=True, stderr=True, stdin=False, tty=False, privileged=False, user='', environment=None, workdir=None, detach_keys=None):
    """
        Sets up an exec instance in a running container.

        Args:
            container (str): Target container where exec instance will be
                created
            cmd (str or list): Command to be executed
            stdout (bool): Attach to stdout. Default: ``True``
            stderr (bool): Attach to stderr. Default: ``True``
            stdin (bool): Attach to stdin. Default: ``False``
            tty (bool): Allocate a pseudo-TTY. Default: False
            privileged (bool): Run as privileged.
            user (str): User to execute command as. Default: root
            environment (dict or list): A dictionary or a list of strings in
                the following format ``["PASSWORD=xxx"]`` or
                ``{"PASSWORD": "xxx"}``.
            workdir (str): Path to working directory for this exec session
            detach_keys (str): Override the key sequence for detaching
                a container. Format is a single character `[a-Z]`
                or `ctrl-<value>` where `<value>` is one of:
                `a-z`, `@`, `^`, `[`, `,` or `_`.
                ~/.docker/config.json is used by default.

        Returns:
            (dict): A dictionary with an exec ``Id`` key.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    if environment is not None and utils.version_lt(self._version, '1.25'):
        raise errors.InvalidVersion('Setting environment for exec is not supported in API < 1.25')
    if isinstance(cmd, str):
        cmd = utils.split_command(cmd)
    if isinstance(environment, dict):
        environment = utils.utils.format_environment(environment)
    data = {'Container': container, 'User': user, 'Privileged': privileged, 'Tty': tty, 'AttachStdin': stdin, 'AttachStdout': stdout, 'AttachStderr': stderr, 'Cmd': cmd, 'Env': environment}
    if workdir is not None:
        if utils.version_lt(self._version, '1.35'):
            raise errors.InvalidVersion('workdir is not supported for API version < 1.35')
        data['WorkingDir'] = workdir
    if detach_keys:
        data['detachKeys'] = detach_keys
    elif 'detachKeys' in self._general_configs:
        data['detachKeys'] = self._general_configs['detachKeys']
    url = self._url('/containers/{0}/exec', container)
    res = self._post_json(url, data=data)
    return self._result(res, True)