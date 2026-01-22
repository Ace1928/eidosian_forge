import copy
import yaml
import json
import os
import socket
import sys
import time
import threading
import logging
import uuid
import warnings
import requests
from packaging.version import Version
from typing import Optional, Dict, Tuple, Type
import ray
import ray._private.services
from ray.autoscaler._private.spark.node_provider import HEAD_NODE_ID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.storage import _load_class
from .utils import (
from .start_hook_base import RayOnSparkStartHook
from .databricks_hook import DefaultDatabricksRayOnSparkStartHook
@PublicAPI(stability='alpha')
def setup_ray_cluster(num_worker_nodes: int, *, num_cpus_worker_node: Optional[int]=None, num_cpus_head_node: Optional[int]=None, num_gpus_worker_node: Optional[int]=None, num_gpus_head_node: Optional[int]=None, object_store_memory_worker_node: Optional[int]=None, object_store_memory_head_node: Optional[int]=None, head_node_options: Optional[Dict]=None, worker_node_options: Optional[Dict]=None, ray_temp_root_dir: Optional[str]=None, strict_mode: bool=False, collect_log_to_path: Optional[str]=None, autoscale: bool=False, autoscale_upscaling_speed: Optional[float]=1.0, autoscale_idle_timeout_minutes: Optional[float]=1.0, **kwargs) -> Tuple[str, str]:
    """
    Set up a ray cluster on the spark cluster by starting a ray head node in the
    spark application's driver side node.
    After creating the head node, a background spark job is created that
    generates an instance of `RayClusterOnSpark` that contains configuration for the
    ray cluster that will run on the Spark cluster's worker nodes.
    After a ray cluster is set up, "RAY_ADDRESS" environment variable is set to
    the cluster address, so you can call `ray.init()` without specifying ray cluster
    address to connect to the cluster. To shut down the cluster you can call
    `ray.util.spark.shutdown_ray_cluster()`.
    Note: If the active ray cluster haven't shut down, you cannot create a new ray
    cluster.

    Args:
        num_worker_nodes: This argument represents how many ray worker nodes to start
            for the ray cluster.
            If autoscale=True, then the ray cluster starts with zero worker node,
            and it can scale up to at most `num_worker_nodes` worker nodes.
            In non-autoscaling mode, you can
            specify the `num_worker_nodes` as `ray.util.spark.MAX_NUM_WORKER_NODES`
            represents a ray cluster
            configuration that will use all available resources configured for the
            spark application.
            To create a spark application that is intended to exclusively run a
            shared ray cluster in non-scaling, it is recommended to set this argument
            to `ray.util.spark.MAX_NUM_WORKER_NODES`.

        num_cpus_worker_node: Number of cpus available to per-ray worker node, if not
            provided, use spark application configuration 'spark.task.cpus' instead.
            **Limitation** Only spark version >= 3.4 or Databricks Runtime 12.x
            supports setting this argument.
        num_cpus_head_node: Number of cpus available to Ray head node, if not provide,
            use 0 instead. Number 0 means tasks requiring CPU resources are not
            scheduled to Ray head node.
        num_gpus_worker_node: Number of gpus available to per-ray worker node, if not
            provided, use spark application configuration
            'spark.task.resource.gpu.amount' instead.
            This argument is only available on spark cluster that is configured with
            'gpu' resources.
            **Limitation** Only spark version >= 3.4 or Databricks Runtime 12.x
            supports setting this argument.
        num_gpus_head_node: Number of gpus available to Ray head node, if not provide,
            use 0 instead.
            This argument is only available on spark cluster which spark driver node
            has GPUs.
        object_store_memory_worker_node: Object store memory available to per-ray worker
            node, but it is capped by
            "dev_shm_available_size * 0.8 / num_tasks_per_spark_worker".
            The default value equals to
            "0.3 * spark_worker_physical_memory * 0.8 / num_tasks_per_spark_worker".
        object_store_memory_head_node: Object store memory available to Ray head
            node, but it is capped by "dev_shm_available_size * 0.8".
            The default value equals to
            "0.3 * spark_driver_physical_memory * 0.8".
        head_node_options: A dict representing Ray head node extra options, these
            options will be passed to `ray start` script. Note you need to convert
            `ray start` options key from `--foo-bar` format to `foo_bar` format.
            For flag options (e.g. '--disable-usage-stats'), you should set the value
            to None in the option dict, like `{"disable_usage_stats": None}`.
            Note: Short name options (e.g. '-v') are not supported.
        worker_node_options: A dict representing Ray worker node extra options,
            these options will be passed to `ray start` script. Note you need to
            convert `ray start` options key from `--foo-bar` format to `foo_bar`
            format.
            For flag options (e.g. '--disable-usage-stats'), you should set the value
            to None in the option dict, like `{"disable_usage_stats": None}`.
            Note: Short name options (e.g. '-v') are not supported.
        ray_temp_root_dir: A local disk path to store the ray temporary data. The
            created cluster will create a subdirectory
            "ray-{head_port}-{random_suffix}" beneath this path.
        strict_mode: Boolean flag to fast-fail initialization of the ray cluster if
            the available spark cluster does not have sufficient resources to fulfill
            the resource allocation for memory, cpu and gpu. When set to true, if the
            requested resources are not available for recommended minimum recommended
            functionality, an exception will be raised that details the inadequate
            spark cluster configuration settings. If overridden as `False`,
            a warning is raised.
        collect_log_to_path: If specified, after ray head / worker nodes terminated,
            collect their logs to the specified path. On Databricks Runtime, we
            recommend you to specify a local path starts with '/dbfs/', because the
            path mounts with a centralized storage device and stored data is persisted
            after Databricks spark cluster terminated.
        autoscale: If True, enable autoscaling, the number of initial Ray worker nodes
            is zero, and the maximum number of Ray worker nodes is set to
            `num_worker_nodes`. Default value is False.
        autoscale_upscaling_speed: If autoscale enabled, it represents the number of
            nodes allowed to be pending as a multiple of the current number of nodes.
            The higher the value, the more aggressive upscaling will be. For example,
            if this is set to 1.0, the cluster can grow in size by at most 100% at any
            time, so if the cluster currently has 20 nodes, at most 20 pending launches
            are allowed. The minimum number of pending launches is 5 regardless of
            this setting.
            Default value is 1.0, minimum value is 1.0
        autoscale_idle_timeout_minutes: If autoscale enabled, it represents the number
            of minutes that need to pass before an idle worker node is removed by the
            autoscaler. The smaller the value, the more aggressive downscaling will be.
            Worker nodes are considered idle when they hold no active tasks, actors,
            or referenced objects (either in-memory or spilled to disk). This parameter
            does not affect the head node.
            Default value is 1.0, minimum value is 0

    Returns:
        A tuple of (address, remote_connection_address)
        "address" is in format of "<ray_head_node_ip>:<port>"
        "remote_connection_address" is in format of
        "ray://<ray_head_node_ip>:<ray-client-server-port>",
        if your client runs on a machine that also hosts a Ray cluster node locally,
        you can connect to the Ray cluster via ``ray.init(address)``,
        otherwise you can connect to the Ray cluster via
        ``ray.init(remote_connection_address)``.
    """
    global _active_ray_cluster
    _check_system_environment()
    head_node_options = head_node_options or {}
    worker_node_options = worker_node_options or {}
    _verify_node_options(head_node_options, _head_node_option_block_keys, 'Ray head node on spark')
    _verify_node_options(worker_node_options, _worker_node_option_block_keys, 'Ray worker node on spark')
    if _active_ray_cluster is not None:
        raise RuntimeError("Current active ray cluster on spark haven't shut down. Please call `ray.util.spark.shutdown_ray_cluster()` before initiating a new Ray cluster on spark.")
    if ray.is_initialized():
        raise RuntimeError('Current python process already initialized Ray, Please shut down it by `ray.shutdown()` before initiating a Ray cluster on spark.')
    spark = get_spark_session()
    spark_master = spark.sparkContext.master
    is_spark_local_mode = spark_master == 'local' or spark_master.startswith('local[')
    if not (spark_master.startswith('spark://') or spark_master.startswith('local-cluster[') or is_spark_local_mode):
        raise RuntimeError('Ray on Spark only supports spark cluster in standalone mode, local-cluster mode or spark local mode.')
    if is_spark_local_mode:
        support_stage_scheduling = False
    elif is_in_databricks_runtime() and Version(os.environ['DATABRICKS_RUNTIME_VERSION']).major >= 12:
        support_stage_scheduling = True
    else:
        import pyspark
        if Version(pyspark.__version__).release >= (3, 4, 0):
            support_stage_scheduling = True
        else:
            support_stage_scheduling = False
    if 'num_cpus_per_node' in kwargs:
        if num_cpus_worker_node is not None:
            raise ValueError("'num_cpus_per_node' and 'num_cpus_worker_node' arguments are equivalent. Only set 'num_cpus_worker_node'.")
        num_cpus_worker_node = kwargs['num_cpus_per_node']
        warnings.warn("'num_cpus_per_node' argument is deprecated, please use 'num_cpus_worker_node' argument instead.", DeprecationWarning)
    if 'num_gpus_per_node' in kwargs:
        if num_gpus_worker_node is not None:
            raise ValueError("'num_gpus_per_node' and 'num_gpus_worker_node' arguments are equivalent. Only set 'num_gpus_worker_node'.")
        num_gpus_worker_node = kwargs['num_gpus_per_node']
        warnings.warn("'num_gpus_per_node' argument is deprecated, please use 'num_gpus_worker_node' argument instead.", DeprecationWarning)
    if 'object_store_memory_per_node' in kwargs:
        if object_store_memory_worker_node is not None:
            raise ValueError("'object_store_memory_per_node' and 'object_store_memory_worker_node' arguments  are equivalent. Only set 'object_store_memory_worker_node'.")
        object_store_memory_worker_node = kwargs['object_store_memory_per_node']
        warnings.warn("'object_store_memory_per_node' argument is deprecated, please use 'object_store_memory_worker_node' argument instead.", DeprecationWarning)
    num_spark_task_cpus = int(spark.sparkContext.getConf().get('spark.task.cpus', '1'))
    if num_cpus_worker_node is not None and num_cpus_worker_node <= 0:
        raise ValueError('Argument `num_cpus_worker_node` value must be > 0.')
    num_spark_task_gpus = int(spark.sparkContext.getConf().get('spark.task.resource.gpu.amount', '0'))
    if num_gpus_worker_node is not None and num_gpus_worker_node < 0:
        raise ValueError('Argument `num_gpus_worker_node` value must be >= 0.')
    if num_cpus_worker_node is not None or num_gpus_worker_node is not None:
        if support_stage_scheduling:
            num_cpus_worker_node = num_cpus_worker_node or num_spark_task_cpus
            num_gpus_worker_node = num_gpus_worker_node or num_spark_task_gpus
            using_stage_scheduling = True
            res_profile = _create_resource_profile(num_cpus_worker_node, num_gpus_worker_node)
        else:
            raise ValueError(f"Current spark version does not support stage scheduling, so that you cannot set the argument `num_cpus_worker_node` and `num_gpus_worker_node` values. Without setting the 2 arguments, per-Ray worker node will be assigned with number of 'spark.task.cpus' (equals to {num_spark_task_cpus}) cpu cores and number of 'spark.task.resource.gpu.amount' (equals to {num_spark_task_gpus}) GPUs. To enable spark stage scheduling, you need to upgrade spark to 3.4 version or use Databricks Runtime 12.x, and you cannot use spark local mode.")
    else:
        using_stage_scheduling = False
        res_profile = None
        num_cpus_worker_node = num_spark_task_cpus
        num_gpus_worker_node = num_spark_task_gpus
    ray_worker_node_heap_mem_bytes, ray_worker_node_object_store_mem_bytes = get_avail_mem_per_ray_worker_node(spark, object_store_memory_worker_node, num_cpus_worker_node, num_gpus_worker_node)
    if num_worker_nodes == MAX_NUM_WORKER_NODES:
        if autoscale:
            raise ValueError('If you set autoscale=True, you cannot set `num_worker_nodes` to `MAX_NUM_WORKER_NODES`, instead, you should set `num_worker_nodes` to the number that represents the upper bound of the ray worker nodes number.')
        num_worker_nodes = get_max_num_concurrent_tasks(spark.sparkContext, res_profile)
    elif num_worker_nodes <= 0:
        raise ValueError("The value of 'num_worker_nodes' argument must be either a positive integer or 'ray.util.spark.MAX_NUM_WORKER_NODES'.")
    insufficient_resources = []
    if num_cpus_worker_node < 4:
        insufficient_resources.append(f"The provided CPU resources for each ray worker are inadequate to start a ray cluster. Based on the total cpu resources available and the configured task sizing, each ray worker node would start with {num_cpus_worker_node} CPU cores. This is less than the recommended value of `4` CPUs per worker. On spark version >= 3.4 or Databricks Runtime 12.x, you can set the argument `num_cpus_worker_node` to a value >= 4 to address it, otherwise you need to increase the spark application configuration 'spark.task.cpus' to a minimum of `4` to address it.")
    if ray_worker_node_heap_mem_bytes < 10 * 1024 * 1024 * 1024:
        insufficient_resources.append(f'The provided memory resources for each ray worker node are inadequate. Based on the total memory available on the spark cluster and the configured task sizing, each ray worker would start with {ray_worker_node_heap_mem_bytes} bytes heap memory. This is less than the recommended value of 10GB. The ray worker node heap memory size is calculated by (SPARK_WORKER_NODE_PHYSICAL_MEMORY / num_local_spark_task_slots * 0.8) - object_store_memory_worker_node. To increase the heap space available, increase the memory in the spark cluster by changing instance types or worker count, reduce the target `num_worker_nodes`, or apply a lower `object_store_memory_worker_node`.')
    if insufficient_resources:
        if strict_mode:
            raise ValueError("You are creating ray cluster on spark with strict mode (it can be disabled by setting argument 'strict_mode=False' when calling API 'setup_ray_cluster'), strict mode requires the spark cluster config satisfying following criterion: \n".join(insufficient_resources))
        else:
            _logger.warning('\n'.join(insufficient_resources))
    if num_cpus_head_node is None:
        num_cpus_head_node = 0
    elif num_cpus_head_node < 0:
        raise ValueError(f'Argument `num_cpus_head_node` value must be >= 0. Current value is {num_cpus_head_node}.')
    if num_gpus_head_node is None:
        num_gpus_head_node = 0
    elif num_gpus_head_node < 0:
        raise ValueError(f'Argument `num_gpus_head_node` value must be >= 0.Current value is {num_gpus_head_node}.')
    if num_cpus_head_node == 0 and num_gpus_head_node == 0 and (object_store_memory_head_node is None):
        heap_memory_head_node = 1024 * 1024 * 1024
        object_store_memory_head_node = 1024 * 1024 * 1024
    else:
        heap_memory_head_node, object_store_memory_head_node = calc_mem_ray_head_node(object_store_memory_head_node)
    with _active_ray_cluster_rwlock:
        cluster = _setup_ray_cluster(num_worker_nodes=num_worker_nodes, num_cpus_worker_node=num_cpus_worker_node, num_cpus_head_node=num_cpus_head_node, num_gpus_worker_node=num_gpus_worker_node, num_gpus_head_node=num_gpus_head_node, using_stage_scheduling=using_stage_scheduling, heap_memory_worker_node=ray_worker_node_heap_mem_bytes, heap_memory_head_node=heap_memory_head_node, object_store_memory_worker_node=ray_worker_node_object_store_mem_bytes, object_store_memory_head_node=object_store_memory_head_node, head_node_options=head_node_options, worker_node_options=worker_node_options, ray_temp_root_dir=ray_temp_root_dir, collect_log_to_path=collect_log_to_path, autoscale=autoscale, autoscale_upscaling_speed=autoscale_upscaling_speed, autoscale_idle_timeout_minutes=autoscale_idle_timeout_minutes)
        cluster.wait_until_ready()
        _active_ray_cluster = cluster
    head_ip = cluster.address.split(':')[0]
    remote_connection_address = f'ray://{head_ip}:{cluster.ray_client_server_port}'
    return (cluster.address, remote_connection_address)