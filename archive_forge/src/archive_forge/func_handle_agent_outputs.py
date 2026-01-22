import os
import pathlib
import tempfile
import uuid
import numpy as np
from ..utils import is_soundfile_availble, is_torch_available, is_vision_available, logging
def handle_agent_outputs(outputs, output_types=None):
    if isinstance(outputs, dict):
        decoded_outputs = {}
        for i, (k, v) in enumerate(outputs.items()):
            if output_types is not None:
                if output_types[i] in AGENT_TYPE_MAPPING:
                    decoded_outputs[k] = AGENT_TYPE_MAPPING[output_types[i]](v)
                else:
                    decoded_outputs[k] = AgentType(v)
            else:
                for _k, _v in INSTANCE_TYPE_MAPPING.items():
                    if isinstance(v, _k):
                        decoded_outputs[k] = _v(v)
                if k not in decoded_outputs:
                    decoded_outputs[k] = AgentType[v]
    elif isinstance(outputs, (list, tuple)):
        decoded_outputs = type(outputs)()
        for i, v in enumerate(outputs):
            if output_types is not None:
                if output_types[i] in AGENT_TYPE_MAPPING:
                    decoded_outputs.append(AGENT_TYPE_MAPPING[output_types[i]](v))
                else:
                    decoded_outputs.append(AgentType(v))
            else:
                found = False
                for _k, _v in INSTANCE_TYPE_MAPPING.items():
                    if isinstance(v, _k):
                        decoded_outputs.append(_v(v))
                        found = True
                if not found:
                    decoded_outputs.append(AgentType(v))
    elif output_types[0] in AGENT_TYPE_MAPPING:
        decoded_outputs = AGENT_TYPE_MAPPING[output_types[0]](outputs)
    else:
        for _k, _v in INSTANCE_TYPE_MAPPING.items():
            if isinstance(outputs, _k):
                return _v(outputs)
        return AgentType(outputs)
    return decoded_outputs