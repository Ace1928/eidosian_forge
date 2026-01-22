from __future__ import annotations
import random
import asyncio
import logging
import contextlib
from enum import Enum
from lazyops.libs.kops.base import *
from lazyops.libs.kops.config import KOpsSettings
from lazyops.libs.kops.utils import cached, DillSerializer, SignalHandler
from lazyops.libs.kops._kopf import kopf
from lazyops.types import lazyproperty
from lazyops.utils import logger
from typing import List, Dict, Union, Any, Optional, Callable, TYPE_CHECKING
import lazyops.libs.kops.types as t
import lazyops.libs.kops.atypes as at
def configure_kopf(cls, _logger: logging.Logger=None, _settings: Optional[kopf.OperatorSettings]=None, enable_event_logging: Optional[bool]=None, event_logging_level: Optional[str]=None, finalizer: Optional[str]=None, storage_prefix: Optional[str]=None, persistent_key: Optional[str]=None, error_delays: Optional[List[int]]=None, startup_functions: Optional[List[Callable]]=None, shutdown_functions: Optional[List[Callable]]=None, kopf_name: Optional[str]=None, app_name: Optional[str]=None, multi_pods: Optional[bool]=True, **kwargs):
    """
        Registers the startup function for configuring kopf.

        Parameters
        """
    enable_event_logging = enable_event_logging if enable_event_logging is not None else cls.settings.kopf_enable_event_logging
    event_logging_level = event_logging_level if event_logging_level is not None else cls.settings.kopf_event_logging_level
    finalizer = finalizer if finalizer is not None else cls.settings.kops_finalizer
    if finalizer != cls.settings.kops_finalizer:
        cls.settings.kops_finalizer = finalizer
    storage_prefix = storage_prefix if storage_prefix is not None else cls.settings.kops_prefix
    persistent_key = persistent_key if persistent_key is not None else cls.settings.kops_persistent_key
    if persistent_key != cls.settings.kops_persistent_key:
        cls.settings.kops_persistent_key = persistent_key
    error_delays = error_delays if error_delays is not None else [10, 20, 30]
    kopf_name = kopf_name if kopf_name is not None else cls.settings.kopf_name
    app_name = app_name if app_name is not None else cls.settings.app_name
    _logger = _logger if _logger is not None else logger
    if startup_functions is not None:
        for func in startup_functions:
            cls.add_function(func, EventType.startup)
    if shutdown_functions is not None:
        for func in shutdown_functions:
            cls.add_function(func, EventType.shutdown)

    @kopf.on.startup()
    async def configure(settings: kopf.OperatorSettings, **kwargs):
        if _settings is not None:
            settings = _settings
        if enable_event_logging is False:
            settings.posting.enabled = enable_event_logging
            _logger.info(f'Kopf Events Enabled: {enable_event_logging}')
        if event_logging_level is not None:
            settings.posting.level = logging.getLevelName(event_logging_level.upper())
            _logger.info(f'Kopf Events Logging Level: {event_logging_level}')
        settings.persistence.finalizer = finalizer
        settings.persistence.progress_storage = kopf.SmartProgressStorage(prefix=storage_prefix)
        settings.persistence.diffbase_storage = kopf.AnnotationsDiffBaseStorage(prefix=storage_prefix, key=persistent_key)
        settings.batching.error_delays = error_delays
        if multi_pods:
            settings.peering.priority = random.randint(0, 32767)
            settings.peering.stealth = True
            if kwargs.get('peering_name'):
                settings.peering.name = kwargs.get('peering_name')
        _logger.info(f'Starting Kopf: {kopf_name} {app_name} @ {cls.settings.build_id}')
        await cls.aset_k8_config()
        if cls._startup_functions:
            _logger.info('Running Startup Functions')
            await cls.run_startup_functions()
        _logger.info('Completed Kopf Startup')

    @kopf.on.login()
    async def login_fn(**kwargs):
        return kopf.login_with_service_account(**kwargs) or kopf.login_with_kubeconfig(**kwargs) if cls.settings.in_k8s else kopf.login_via_client(**kwargs)