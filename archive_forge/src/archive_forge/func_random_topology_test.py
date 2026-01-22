import torch
import numpy as np
import argparse
from typing import Dict
def random_topology_test(seed, *inp_tensor_list):
    np.random.seed(int(seed.numpy().tolist()))
    tensor_list = [*inp_tensor_list]
    num_tensor = len(tensor_list)
    num_const = np.random.randint(0, num_tensor + 1)
    const_list = np.random.random(num_const)
    if DEBUG_PRINT:
        for const_item in const_list:
            print('----- real number {:.10f}', const_item)

    def get_root(x, dependency_map):
        if x in dependency_map:
            return get_root(dependency_map[x], dependency_map)
        else:
            return x
    d_map: Dict[int, int] = {}
    num_sets = num_tensor
    candidate = list(range(num_tensor))
    unary_operations = [torch.sigmoid, torch.relu]
    binary_operations = [torch.add, torch.sub, torch.mul]
    u_op_size = len(unary_operations)
    b_op_size = len(binary_operations)
    num_operations = np.random.randint(num_sets - 1, num_sets * GRAPH_FACTOR)
    ret_list = []
    while num_operations >= 0 or num_sets > 1:
        index = np.random.randint(0, len(candidate))
        op_index = np.random.randint(0, u_op_size + b_op_size)
        lh_index = candidate[index]
        rh_index = None
        out_tensor = None
        if DEBUG_PRINT:
            print('iteration {}, num_sets{}, candidates {}, tensor_list {}, lh_index {}, op_index {}'.format(num_operations, num_sets, candidate, len(tensor_list), lh_index, op_index))
        if num_operations >= 0:
            num_operations -= 1
            if op_index < u_op_size:
                out_tensor = unary_operations[op_index](tensor_list[lh_index])
            else:
                op_2_index = np.random.randint(0, len(tensor_list) + num_const)
                if op_2_index < len(tensor_list):
                    if op_2_index == lh_index:
                        op_2_index = (op_2_index + 1) % len(tensor_list)
                    rh_index = op_2_index
                else:
                    left = tensor_list[lh_index]
                    right = const_list[op_2_index - len(tensor_list)]
                    out_tensor = binary_operations[op_index - u_op_size](left, right)
                if DEBUG_PRINT:
                    print(f'binary, op_2_index {op_2_index}, rh_index ?{rh_index}')
        else:
            cand_index = np.random.randint(0, len(candidate))
            if cand_index == index:
                cand_index = (cand_index + 1) % len(candidate)
            rh_index = candidate[cand_index]
            if DEBUG_PRINT:
                print(f'binary rh_index ?{rh_index}')
        candidate[index] = len(tensor_list)
        lh_root = get_root(lh_index, d_map)
        if rh_index is not None:
            out_tensor = binary_operations[op_index - u_op_size](tensor_list[lh_index], tensor_list[rh_index])
            if rh_index in candidate:
                candidate.remove(rh_index)
            rh_root = get_root(rh_index, d_map)
            if lh_root != rh_root:
                num_sets -= 1
                d_map[rh_root] = len(tensor_list)
        d_map[lh_root] = len(tensor_list)
        tensor_list.append(out_tensor)
    for ind in candidate:
        ret_list.append(tensor_list[ind])
    out_list = np.random.choice(range(num_tensor, len(tensor_list)), np.random.randint(0, len(tensor_list) - num_tensor), False)
    for ind in out_list:
        if ind not in candidate:
            ret_list.append(tensor_list[ind])
    if DEBUG_PRINT:
        print(f'ended with tensor_list: {len(tensor_list)}')
    return tuple(ret_list)