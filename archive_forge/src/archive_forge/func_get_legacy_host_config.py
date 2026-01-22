from __future__ import annotations
import argparse
import collections.abc as c
import dataclasses
import enum
import os
import types
import typing as t
from ..constants import (
from ..util import (
from ..docker_util import (
from ..completion import (
from ..host_configs import (
from ..data import (
def get_legacy_host_config(mode: TargetMode, options: LegacyHostOptions) -> tuple[ControllerHostConfig, list[HostConfig], t.Optional[FallbackDetail]]:
    """
    Returns controller and target host configs derived from the provided legacy host options.
    The goal is to match the original behavior, by using non-split testing whenever possible.
    When the options support the controller, use the options for the controller and use ControllerConfig for the targets.
    When the options do not support the controller, use the options for the targets and use a default controller config influenced by the options.
    """
    venv_fallback = 'venv/default'
    docker_fallback = 'default'
    remote_fallback = get_fallback_remote_controller()
    controller_fallback: t.Optional[tuple[str, str, FallbackReason]] = None
    controller: t.Optional[ControllerHostConfig]
    targets: list[HostConfig]
    if options.venv:
        if controller_python(options.python) or not options.python:
            controller = OriginConfig(python=VirtualPythonConfig(version=options.python or 'default', system_site_packages=options.venv_system_site_packages))
        else:
            controller_fallback = (f'origin:python={venv_fallback}', f'--venv --python {options.python}', FallbackReason.PYTHON)
            controller = OriginConfig(python=VirtualPythonConfig(version='default', system_site_packages=options.venv_system_site_packages))
        if mode in (TargetMode.SANITY, TargetMode.UNITS):
            python = native_python(options)
            if python:
                control_targets = [ControllerConfig(python=python)]
            else:
                control_targets = controller.get_default_targets(HostContext(controller_config=controller))
            if mode == TargetMode.UNITS:
                targets = [ControllerConfig(python=VirtualPythonConfig(version=target.python.version, path=target.python.path, system_site_packages=options.venv_system_site_packages)) for target in control_targets]
            else:
                targets = t.cast(list[HostConfig], control_targets)
        else:
            targets = [ControllerConfig(python=VirtualPythonConfig(version=options.python or 'default', system_site_packages=options.venv_system_site_packages))]
    elif options.docker:
        docker_config = filter_completion(docker_completion()).get(options.docker)
        if docker_config:
            if options.python and options.python not in docker_config.supported_pythons:
                raise PythonVersionUnsupportedError(f'--docker {options.docker}', options.python, docker_config.supported_pythons)
            if docker_config.controller_supported:
                if controller_python(options.python) or not options.python:
                    controller = DockerConfig(name=options.docker, python=native_python(options), privileged=options.docker_privileged, seccomp=options.docker_seccomp, memory=options.docker_memory)
                    targets = controller_targets(mode, options, controller)
                else:
                    controller_fallback = (f'docker:{options.docker}', f'--docker {options.docker} --python {options.python}', FallbackReason.PYTHON)
                    controller = DockerConfig(name=options.docker)
                    targets = controller_targets(mode, options, controller)
            else:
                controller_fallback = (f'docker:{docker_fallback}', f'--docker {options.docker}', FallbackReason.ENVIRONMENT)
                controller = DockerConfig(name=docker_fallback)
                targets = [DockerConfig(name=options.docker, python=native_python(options), privileged=options.docker_privileged, seccomp=options.docker_seccomp, memory=options.docker_memory)]
        else:
            if not options.python:
                raise PythonVersionUnspecifiedError(f'--docker {options.docker}')
            if controller_python(options.python):
                controller = DockerConfig(name=options.docker, python=native_python(options), privileged=options.docker_privileged, seccomp=options.docker_seccomp, memory=options.docker_memory)
                targets = controller_targets(mode, options, controller)
            else:
                controller_fallback = (f'docker:{docker_fallback}', f'--docker {options.docker} --python {options.python}', FallbackReason.PYTHON)
                controller = DockerConfig(name=docker_fallback)
                targets = [DockerConfig(name=options.docker, python=native_python(options), privileged=options.docker_privileged, seccomp=options.docker_seccomp, memory=options.docker_memory)]
    elif options.remote:
        remote_config = filter_completion(remote_completion()).get(options.remote)
        context, reason = (None, None)
        if remote_config:
            if options.python and options.python not in remote_config.supported_pythons:
                raise PythonVersionUnsupportedError(f'--remote {options.remote}', options.python, remote_config.supported_pythons)
            if remote_config.controller_supported:
                if controller_python(options.python) or not options.python:
                    controller = PosixRemoteConfig(name=options.remote, python=native_python(options), provider=options.remote_provider, arch=options.remote_arch)
                    targets = controller_targets(mode, options, controller)
                else:
                    controller_fallback = (f'remote:{options.remote}', f'--remote {options.remote} --python {options.python}', FallbackReason.PYTHON)
                    controller = PosixRemoteConfig(name=options.remote, provider=options.remote_provider, arch=options.remote_arch)
                    targets = controller_targets(mode, options, controller)
            else:
                context, reason = (f'--remote {options.remote}', FallbackReason.ENVIRONMENT)
                controller = None
                targets = [PosixRemoteConfig(name=options.remote, python=native_python(options), provider=options.remote_provider, arch=options.remote_arch)]
        elif mode == TargetMode.SHELL and options.remote.startswith('windows/'):
            if options.python and options.python not in CONTROLLER_PYTHON_VERSIONS:
                raise ControllerNotSupportedError(f'--python {options.python}')
            controller = OriginConfig(python=native_python(options))
            targets = [WindowsRemoteConfig(name=options.remote, provider=options.remote_provider, arch=options.remote_arch)]
        else:
            if not options.python:
                raise PythonVersionUnspecifiedError(f'--remote {options.remote}')
            if controller_python(options.python):
                controller = PosixRemoteConfig(name=options.remote, python=native_python(options), provider=options.remote_provider, arch=options.remote_arch)
                targets = controller_targets(mode, options, controller)
            else:
                context, reason = (f'--remote {options.remote} --python {options.python}', FallbackReason.PYTHON)
                controller = None
                targets = [PosixRemoteConfig(name=options.remote, python=native_python(options), provider=options.remote_provider, arch=options.remote_arch)]
        if not controller:
            if docker_available():
                controller_fallback = (f'docker:{docker_fallback}', context, reason)
                controller = DockerConfig(name=docker_fallback)
            else:
                controller_fallback = (f'remote:{remote_fallback}', context, reason)
                controller = PosixRemoteConfig(name=remote_fallback)
    elif controller_python(options.python) or not options.python:
        controller = OriginConfig(python=native_python(options))
        targets = controller_targets(mode, options, controller)
    else:
        controller_fallback = ('origin:python=default', f'--python {options.python}', FallbackReason.PYTHON)
        controller = OriginConfig()
        targets = controller_targets(mode, options, controller)
    if controller_fallback:
        controller_option, context, reason = controller_fallback
        if mode.no_fallback:
            raise ControllerNotSupportedError(context)
        fallback_detail = FallbackDetail(reason=reason, message=f'Using `--controller {controller_option}` since `{context}` does not support the controller.')
    else:
        fallback_detail = None
    if mode.one_host and any((not isinstance(target, ControllerConfig) for target in targets)):
        raise ControllerNotSupportedError(controller_fallback[1])
    if mode == TargetMode.NO_TARGETS:
        targets = []
    else:
        targets = handle_non_posix_targets(mode, options, targets)
    return (controller, targets, fallback_detail)