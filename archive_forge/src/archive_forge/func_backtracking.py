from dataclasses import dataclass, field
from collections import defaultdict
import copy
import torch
from torch.fx import (
from torch.fx._compatibility import compatibility
from typing import Dict, List, Set, Any, Union, Tuple
import logging
import os
def backtracking(anchor_index, match):
    if anchor_index == len(match_candidates_list):
        match.placeholder_nodes = [match.nodes_map[pn] for pn in self.pattern_placeholder_nodes]
        match.returning_nodes = [match.nodes_map[pn] for pn in self.pattern_returning_nodes]
        matches.append(match)
        logger.info('Found a match: %s\n', match)
        return
    pattern_anchor, candidate_nodes = match_candidates_list[anchor_index]
    saved_match = copy.copy(match)
    for node in candidate_nodes:
        logger.info('Trying to match anchor %s to %s', pattern_anchor, node)
        match_found = self._match_nodes(pattern_anchor, node, match)
        if match_found:
            backtracking(anchor_index + 1, match)
        else:
            logger.info('Failed to match anchor %s to %s\n', pattern_anchor, node)
        match = copy.copy(saved_match)