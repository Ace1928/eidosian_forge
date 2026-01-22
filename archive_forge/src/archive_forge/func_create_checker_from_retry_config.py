import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
def create_checker_from_retry_config(config, operation_name=None):
    checkers = []
    max_attempts = None
    retryable_exceptions = []
    if '__default__' in config:
        policies = config['__default__'].get('policies', [])
        max_attempts = config['__default__']['max_attempts']
        for key in policies:
            current_config = policies[key]
            checkers.append(_create_single_checker(current_config))
            retry_exception = _extract_retryable_exception(current_config)
            if retry_exception is not None:
                retryable_exceptions.extend(retry_exception)
    if operation_name is not None and config.get(operation_name) is not None:
        operation_policies = config[operation_name]['policies']
        for key in operation_policies:
            checkers.append(_create_single_checker(operation_policies[key]))
            retry_exception = _extract_retryable_exception(operation_policies[key])
            if retry_exception is not None:
                retryable_exceptions.extend(retry_exception)
    if len(checkers) == 1:
        return MaxAttemptsDecorator(checkers[0], max_attempts=max_attempts)
    else:
        multi_checker = MultiChecker(checkers)
        return MaxAttemptsDecorator(multi_checker, max_attempts=max_attempts, retryable_exceptions=tuple(retryable_exceptions))