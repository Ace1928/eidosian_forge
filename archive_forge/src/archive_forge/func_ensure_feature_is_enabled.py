from __future__ import absolute_import, division, print_function
import json
import re
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native
def ensure_feature_is_enabled(client, feature_str):
    enabled_features = client.get_enabled_features()
    if enabled_features is None:
        enabled_features = []
    if feature_str not in enabled_features:
        client.enable_features(feature_str)
        client.save_config()