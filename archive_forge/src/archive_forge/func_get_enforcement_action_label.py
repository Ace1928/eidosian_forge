from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict
from apitools.base.protorpclite import messages
def get_enforcement_action_label(enforcement_action):
    return ENFORCEMENT_ACTION_LABEL_MAP.get(enforcement_action, 'ENFORCEMENT_ACTION_UNSPECIFIED')