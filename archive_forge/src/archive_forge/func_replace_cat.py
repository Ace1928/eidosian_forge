import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def replace_cat(self, graph: torch.fx.GraphModule, split_node: torch.fx.Node, next_users: List[torch.fx.Node], user_inputs_list_new, transform_params_list: List[List[_TransformParam]]):
    split_dim = split_node.kwargs['dim']
    split_users = split_node.users.keys()
    new_cats = []
    for user_node, user_inputs_new, transform_params in zip(next_users, user_inputs_list_new, transform_params_list):
        if user_node.target not in {torch.cat, torch.stack}:
            next_cat_input = 0
            for input_node in user_node.all_input_nodes:
                if input_node in split_users:
                    user_node.replace_input_with(input_node, user_inputs_new[next_cat_input])
                    next_cat_input += 1
            continue
        cat_dim = get_arg_value(user_node, 1, 'dim')
        user_inputs_new_transformed = []
        to_stack = []
        stack_dim = None
        with graph.inserting_before(user_node):
            for user_input_new, transform_param in zip(user_inputs_new, transform_params):
                unflatten_params, movedim_params, unsqueeze_params, flatten_params = transform_param
                if unsqueeze_params and (stack_dim is None or stack_dim == unsqueeze_params[0]):
                    to_stack.append(user_input_new)
                    stack_dim = unsqueeze_params[0]
                    continue
                elif to_stack:
                    stacked_input = graph.call_function(torch.stack, args=(to_stack,), kwargs={'dim': stack_dim})
                    to_stack = []
                    stack_dim = None
                    user_inputs_new_transformed.append(stacked_input)
                    if unsqueeze_params:
                        to_stack.append(user_input_new)
                        stack_dim = unsqueeze_params[0]
                        continue
                if unflatten_params:
                    user_input_new = graph.call_function(torch.unflatten, args=(user_input_new, *unflatten_params))
                if movedim_params:
                    user_input_new = graph.call_function(torch.movedim, args=(user_input_new, *movedim_params))
                if flatten_params:
                    user_input_new = graph.call_function(torch.flatten, args=(user_input_new, *flatten_params))
                user_inputs_new_transformed.append(user_input_new)
            if to_stack:
                stacked_input = graph.call_function(torch.stack, args=(to_stack,), kwargs={'dim': stack_dim})
                user_inputs_new_transformed.append(stacked_input)
        with graph.inserting_after(user_node):
            if len(user_inputs_new_transformed) > 1:
                new_cat_node = graph.call_function(torch.cat, args=(user_inputs_new_transformed,), kwargs={'dim': cat_dim})
                new_cat_node.meta.update(user_node.meta)
                counters['inductor']['scmerge_cat_added'] += 1
            else:
                new_cat_node = user_inputs_new_transformed[-1]
        if user_node.target == torch.cat and split_dim != cat_dim and (split_node.target == torch.split):
            with graph.inserting_after(new_cat_node):
                new_cat_node = graph.call_function(torch.flatten, args=(new_cat_node, cat_dim, cat_dim + 1))
        user_node.replace_all_uses_with(new_cat_node)
        new_cats.append(new_cat_node)