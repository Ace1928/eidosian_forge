import collections.abc
import json
import typing as ty
from urllib import parse
from urllib import request
from openstack import exceptions
from openstack.orchestration.util import environment_format
from openstack.orchestration.util import template_format
from openstack.orchestration.util import utils
def process_multiple_environments_and_files(env_paths=None, template=None, template_url=None, env_path_is_object=None, object_request=None, env_list_tracker=None):
    """Reads one or more environment files.

    Reads in each specified environment file and returns a dictionary
    of the filenames->contents (suitable for the files dict)
    and the consolidated environment (after having applied the correct
    overrides based on order).

    If a list is provided in the env_list_tracker parameter, the behavior
    is altered to take advantage of server-side environment resolution.
    Specifically, this means:

    * Populating env_list_tracker with an ordered list of environment file
      URLs to be passed to the server
    * Including the contents of each environment file in the returned
      files dict, keyed by one of the URLs in env_list_tracker

    :param env_paths: list of paths to the environment files to load; if
           None, empty results will be returned
    :type  env_paths: list or None
    :param template: unused; only included for API compatibility
    :param template_url: unused; only included for API compatibility
    :param env_list_tracker: if specified, environment filenames will be
           stored within
    :type  env_list_tracker: list or None
    :return: tuple of files dict and a dict of the consolidated environment
    :rtype:  tuple
    """
    merged_files: ty.Dict[str, str] = {}
    merged_env: ty.Dict[str, ty.Dict] = {}
    include_env_in_files = env_list_tracker is not None
    if env_paths:
        for env_path in env_paths:
            files, env = process_environment_and_files(env_path=env_path, template=template, template_url=template_url, env_path_is_object=env_path_is_object, object_request=object_request, include_env_in_files=include_env_in_files)
            merged_files.update(files)
            merged_env = deep_update(merged_env, env)
            if env_list_tracker is not None:
                env_url = utils.normalise_file_path_to_url(env_path)
                env_list_tracker.append(env_url)
    return (merged_files, merged_env)