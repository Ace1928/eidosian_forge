import hashlib
import importlib.util
import os
import re
import subprocess
import tempfile
import yaml
import ray
def load_package(config_path: str) -> '_RuntimePackage':
    """Load the code package given its config path.

    Args:
        config_path: The path to the configuration YAML that defines
            the package. For documentation on the packaging format, see the
            example YAML in ``example_pkg/ray_pkg.yaml``.

    Examples:

        .. code-block :: python

            # Load from local.
            my_pkg = load_package("~/path/to/my_pkg.yaml")

            # Load from GitHub.
            my_pkg = ray.util.load_package(
            "https://raw.githubusercontent.com/user/repo/refspec"
            "/path/to/package/my_pkg.yaml")

            # Inspect the package runtime env.
            print(my_pkg._runtime_env)
            {"conda": {...},
            "docker": "anyscale-ml/ray-ml:nightly-py38-cpu",
            "working_dir": "https://github.com/demo/foo/blob/v3.0/project/"}

            # Run remote functions from the package.
            my_pkg.my_func.remote(1, 2)

            # Create actors from the package.
            actor = my_pkg.MyActor.remote(3, 4)

            # Create new remote funcs in the same env as a package.
            @ray.remote(runtime_env=my_pkg._runtime_env)
            def f(): ...
    """
    from ray._private.runtime_env.packaging import get_uri_for_directory, upload_package_if_needed
    config_path = _download_from_github_if_needed(config_path)
    if not os.path.exists(config_path):
        raise ValueError('Config file does not exist: {}'.format(config_path))
    config = yaml.safe_load(open(config_path).read())
    base_dir = os.path.abspath(os.path.dirname(config_path))
    runtime_env = config['runtime_env']
    if 'working_dir' not in runtime_env:
        pkg_uri = get_uri_for_directory(base_dir, excludes=[])

        def do_register_package():
            upload_package_if_needed(pkg_uri, _pkg_tmp(), base_dir)
        if ray.is_initialized():
            do_register_package()
        else:
            ray._private.worker._post_init_hooks.append(do_register_package)
        runtime_env['working_dir'] = pkg_uri
    conda_yaml = os.path.join(base_dir, 'conda.yaml')
    if os.path.exists(conda_yaml):
        if 'conda' in runtime_env:
            raise ValueError('Both conda.yaml and conda: section found in package')
        runtime_env['conda'] = yaml.safe_load(open(conda_yaml).read())
    pkg = _RuntimePackage(name=config['name'], desc=config['description'], interface_file=os.path.join(base_dir, config['interface_file']), runtime_env=runtime_env)
    return pkg