from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
def remove_duplicate_context(cmds):
    """Helper method to remove duplicate telemetry context commands"""
    if not cmds:
        return cmds
    feature_indices = [i for i, x in enumerate(cmds) if x == 'feature telemetry']
    telemetry_indices = [i for i, x in enumerate(cmds) if x == 'telemetry']
    if len(feature_indices) == 1 and len(telemetry_indices) == 1:
        return cmds
    if len(feature_indices) == 1 and (not telemetry_indices):
        return cmds
    if len(telemetry_indices) == 1 and (not feature_indices):
        return cmds
    if feature_indices and feature_indices[-1] > 1:
        cmds.pop(feature_indices[-1])
        return remove_duplicate_context(cmds)
    if telemetry_indices and telemetry_indices[-1] > 1:
        cmds.pop(telemetry_indices[-1])
        return remove_duplicate_context(cmds)