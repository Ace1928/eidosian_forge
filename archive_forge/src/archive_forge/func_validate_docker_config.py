from pathlib import Path
from typing import Any, Dict
from ray.autoscaler._private.cli_logger import cli_logger
def validate_docker_config(config: Dict[str, Any]) -> None:
    """Checks whether the Docker configuration is valid."""
    if 'docker' not in config:
        return
    _check_docker_file_mounts(config.get('file_mounts', {}))
    docker_image = config['docker'].get('image')
    cname = config['docker'].get('container_name')
    head_docker_image = config['docker'].get('head_image', docker_image)
    worker_docker_image = config['docker'].get('worker_image', docker_image)
    image_present = docker_image or (head_docker_image and worker_docker_image)
    if not cname and (not image_present):
        return
    else:
        assert cname and image_present, 'Must provide a container & image name'
    return None