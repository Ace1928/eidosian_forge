from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import multiprocessing
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def detect_and_set_best_config(is_estimated_multi_file_workload):
    """Determines best app config based on system and workload."""
    if is_estimated_multi_file_workload:
        _set_if_not_user_set('sliced_object_download_component_size', COMPONENT_SIZE)
        _set_if_not_user_set('sliced_object_download_max_components', MULTI_FILE_SLICED_OBJECT_DOWNLOAD_MAX_COMPONENTS)
        if multiprocessing.cpu_count() < 4:
            log.info('Using low CPU count, multi-file workload config.')
            _set_if_not_user_set('process_count', MULTI_FILE_LOW_CPU_PROCESS_COUNT)
            _set_if_not_user_set('thread_count', MULTI_FILE_LOW_CPU_THREAD_COUNT)
            _set_if_not_user_set('sliced_object_download_threshold', MULTI_FILE_LOW_CPU_SLICED_OBJECT_DOWNLOAD_THRESHOLD)
        else:
            log.info('Using high CPU count, multi-file workload config.')
            _set_if_not_user_set('process_count', MULTI_FILE_HIGH_CPU_PROCESS_COUNT)
            _set_if_not_user_set('thread_count', MULTI_FILE_HIGH_CPU_THREAD_COUNT)
            _set_if_not_user_set('sliced_object_download_threshold', MULTI_FILE_HIGH_CPU_SLICED_OBJECT_DOWNLOAD_THRESHOLD)
    else:
        _set_if_not_user_set('sliced_object_download_threshold', SINGLE_FILE_SLICED_OBJECT_DOWNLOAD_THRESHOLD)
        _set_if_not_user_set('sliced_object_download_component_size', COMPONENT_SIZE)
        if multiprocessing.cpu_count() < 8:
            log.info('Using low CPU count, single-file workload config.')
            _set_if_not_user_set('process_count', SINGLE_FILE_LOW_CPU_PROCESS_COUNT)
            _set_if_not_user_set('thread_count', SINGLE_FILE_THREAD_COUNT)
            _set_if_not_user_set('sliced_object_download_max_components', SINGLE_FILE_LOW_CPU_SLICED_OBJECT_DOWNLOAD_MAX_COMPONENTS)
        else:
            log.info('Using high CPU count, single-file workload config.')
            _set_if_not_user_set('process_count', SINGLE_FILE_HIGH_CPU_PROCESS_COUNT)
            _set_if_not_user_set('thread_count', SINGLE_FILE_THREAD_COUNT)
            _set_if_not_user_set('sliced_object_download_max_components', SINGLE_FILE_HIGH_CPU_SLICED_OBJECT_DOWNLOAD_MAX_COMPONENTS)