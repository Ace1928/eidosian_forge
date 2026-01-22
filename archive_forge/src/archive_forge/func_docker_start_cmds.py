from pathlib import Path
from typing import Any, Dict
from ray.autoscaler._private.cli_logger import cli_logger
def docker_start_cmds(user, image, mount_dict, container_name, user_options, cluster_name, home_directory, docker_cmd):
    from ray.autoscaler.sdk import get_docker_host_mount_location
    docker_mount_prefix = get_docker_host_mount_location(cluster_name)
    mount = {f'{docker_mount_prefix}/{dst}': dst for dst in mount_dict}
    mount_flags = ' '.join(['-v {src}:{dest}'.format(src=k, dest=v.replace('~/', home_directory + '/')) for k, v in mount.items()])
    env_vars = {'LC_ALL': 'C.UTF-8', 'LANG': 'C.UTF-8'}
    env_flags = ' '.join(['-e {name}={val}'.format(name=k, val=v) for k, v in env_vars.items()])
    user_options_str = ' '.join(user_options)
    docker_run = [docker_cmd, 'run', '--rm', '--name {}'.format(container_name), '-d', '-it', mount_flags, env_flags, user_options_str, '--net=host', image, 'bash']
    return ' '.join(docker_run)