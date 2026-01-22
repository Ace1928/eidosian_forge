import json
import logging
import os
import struct
from typing import Any, List, Optional
import torch
import numpy as np
from google.protobuf import struct_pb2
from tensorboard.compat.proto.summary_pb2 import (
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.pr_curve.plugin_data_pb2 import PrCurvePluginData
from tensorboard.plugins.text.plugin_data_pb2 import TextPluginData
from ._convert_np import make_np
from ._utils import _prepare_video, convert_to_HWC
def hparams(hparam_dict=None, metric_dict=None, hparam_domain_discrete=None):
    """Output three `Summary` protocol buffers needed by hparams plugin.

    `Experiment` keeps the metadata of an experiment, such as the name of the
      hyperparameters and the name of the metrics.
    `SessionStartInfo` keeps key-value pairs of the hyperparameters
    `SessionEndInfo` describes status of the experiment e.g. STATUS_SUCCESS

    Args:
      hparam_dict: A dictionary that contains names of the hyperparameters
        and their values.
      metric_dict: A dictionary that contains names of the metrics
        and their values.
      hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that
        contains names of the hyperparameters and all discrete values they can hold

    Returns:
      The `Summary` protobufs for Experiment, SessionStartInfo and
        SessionEndInfo
    """
    import torch
    from tensorboard.plugins.hparams.api_pb2 import DataType, Experiment, HParamInfo, MetricInfo, MetricName, Status
    from tensorboard.plugins.hparams.metadata import EXPERIMENT_TAG, PLUGIN_DATA_VERSION, PLUGIN_NAME, SESSION_END_INFO_TAG, SESSION_START_INFO_TAG
    from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData, SessionEndInfo, SessionStartInfo
    if not isinstance(hparam_dict, dict):
        logger.warning('parameter: hparam_dict should be a dictionary, nothing logged.')
        raise TypeError('parameter: hparam_dict should be a dictionary, nothing logged.')
    if not isinstance(metric_dict, dict):
        logger.warning('parameter: metric_dict should be a dictionary, nothing logged.')
        raise TypeError('parameter: metric_dict should be a dictionary, nothing logged.')
    hparam_domain_discrete = hparam_domain_discrete or {}
    if not isinstance(hparam_domain_discrete, dict):
        raise TypeError('parameter: hparam_domain_discrete should be a dictionary, nothing logged.')
    for k, v in hparam_domain_discrete.items():
        if k not in hparam_dict or not isinstance(v, list) or (not all((isinstance(d, type(hparam_dict[k])) for d in v))):
            raise TypeError(f'parameter: hparam_domain_discrete[{k}] should be a list of same type as hparam_dict[{k}].')
    hps = []
    ssi = SessionStartInfo()
    for k, v in hparam_dict.items():
        if v is None:
            continue
        if isinstance(v, (int, float)):
            ssi.hparams[k].number_value = v
            if k in hparam_domain_discrete:
                domain_discrete: Optional[struct_pb2.ListValue] = struct_pb2.ListValue(values=[struct_pb2.Value(number_value=d) for d in hparam_domain_discrete[k]])
            else:
                domain_discrete = None
            hps.append(HParamInfo(name=k, type=DataType.Value('DATA_TYPE_FLOAT64'), domain_discrete=domain_discrete))
            continue
        if isinstance(v, str):
            ssi.hparams[k].string_value = v
            if k in hparam_domain_discrete:
                domain_discrete = struct_pb2.ListValue(values=[struct_pb2.Value(string_value=d) for d in hparam_domain_discrete[k]])
            else:
                domain_discrete = None
            hps.append(HParamInfo(name=k, type=DataType.Value('DATA_TYPE_STRING'), domain_discrete=domain_discrete))
            continue
        if isinstance(v, bool):
            ssi.hparams[k].bool_value = v
            if k in hparam_domain_discrete:
                domain_discrete = struct_pb2.ListValue(values=[struct_pb2.Value(bool_value=d) for d in hparam_domain_discrete[k]])
            else:
                domain_discrete = None
            hps.append(HParamInfo(name=k, type=DataType.Value('DATA_TYPE_BOOL'), domain_discrete=domain_discrete))
            continue
        if isinstance(v, torch.Tensor):
            v = make_np(v)[0]
            ssi.hparams[k].number_value = v
            hps.append(HParamInfo(name=k, type=DataType.Value('DATA_TYPE_FLOAT64')))
            continue
        raise ValueError('value should be one of int, float, str, bool, or torch.Tensor')
    content = HParamsPluginData(session_start_info=ssi, version=PLUGIN_DATA_VERSION)
    smd = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name=PLUGIN_NAME, content=content.SerializeToString()))
    ssi = Summary(value=[Summary.Value(tag=SESSION_START_INFO_TAG, metadata=smd)])
    mts = [MetricInfo(name=MetricName(tag=k)) for k in metric_dict.keys()]
    exp = Experiment(hparam_infos=hps, metric_infos=mts)
    content = HParamsPluginData(experiment=exp, version=PLUGIN_DATA_VERSION)
    smd = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name=PLUGIN_NAME, content=content.SerializeToString()))
    exp = Summary(value=[Summary.Value(tag=EXPERIMENT_TAG, metadata=smd)])
    sei = SessionEndInfo(status=Status.Value('STATUS_SUCCESS'))
    content = HParamsPluginData(session_end_info=sei, version=PLUGIN_DATA_VERSION)
    smd = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name=PLUGIN_NAME, content=content.SerializeToString()))
    sei = Summary(value=[Summary.Value(tag=SESSION_END_INFO_TAG, metadata=smd)])
    return (exp, ssi, sei)