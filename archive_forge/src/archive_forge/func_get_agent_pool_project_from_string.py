from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import properties
def get_agent_pool_project_from_string(agent_pool_string):
    prefix_search_result = re.search(_AGENT_POOLS_PREFIX_REGEX, agent_pool_string)
    if prefix_search_result:
        return prefix_search_result.group(1)
    raise ValueError('Full agent pool prefix required to extract project from string (e.g. "projects/[project ID]/agentPools/[pool name]).')