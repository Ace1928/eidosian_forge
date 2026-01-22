from pathlib import Path
from typing import Any, Dict
from ray.autoscaler._private.cli_logger import cli_logger
def with_docker_exec(cmds, container_name, docker_cmd, env_vars=None, with_interactive=False):
    assert docker_cmd, 'Must provide docker command'
    env_str = ''
    if env_vars:
        env_str = ' '.join(['-e {env}=${env}'.format(env=env) for env in env_vars])
    return ['docker exec {interactive} {env} {container} /bin/bash -c {cmd} '.format(interactive='-it' if with_interactive else '', env=env_str, container=container_name, cmd=quote(cmd)) for cmd in cmds]