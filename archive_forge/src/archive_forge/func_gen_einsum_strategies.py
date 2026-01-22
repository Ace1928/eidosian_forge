import itertools
from dataclasses import dataclass
from typing import List, Tuple
from torch.distributed._tensor.op_schema import OpStrategy, PlacementStrategy
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def gen_einsum_strategies(equation: str, mesh: DeviceMesh, *, linearity: bool=False) -> OpStrategy:
    """
    Generate a strategy list for the ops that follow einsum style notation.
    """
    input_dims, output_dim = EinsumDims.parse_equation(equation)
    edims = EinsumDims.parse_dims(input_dims, output_dim)
    all_mesh_dim_strategies = []
    for mesh_dim in range(mesh.ndim):
        mesh_dim_strategies = []
        placement_list: List[Placement] = [Replicate()] * (len(input_dims) + 1)
        mesh_dim_strategies.append(placement_list)
        if mesh.size(mesh_dim) <= 1:
            continue
        for batch_dim in edims.batch_dims:
            output_batch_dim = output_dim.index(batch_dim)
            placement_list = [Shard(output_batch_dim)]
            for input_dim in input_dims:
                input_batch_dim = input_dim.index(batch_dim)
                placement_list.append(Shard(input_batch_dim))
            mesh_dim_strategies.append(placement_list)
        for contracting_dim in edims.contracting_dims:
            placement_list = [_Partial()]
            for input_dim in input_dims:
                input_contracting_dim = input_dim.index(contracting_dim)
                placement_list.append(Shard(input_contracting_dim))
            mesh_dim_strategies.append(placement_list)
        for lhs_dim in edims.lhs_out_only_dims:
            lhs_free_dim = output_dim.index(lhs_dim)
            lhs_placement_list: List[Placement] = [Shard(lhs_free_dim), Shard(lhs_free_dim), Replicate()]
            mesh_dim_strategies.append(lhs_placement_list)
        for rhs_dim in edims.rhs_out_only_dims:
            rhs_free_dim = output_dim.index(rhs_dim)
            rhs_placement_list: List[Placement] = [Shard(rhs_free_dim), Replicate(), Shard(rhs_free_dim)]
            mesh_dim_strategies.append(rhs_placement_list)
        if linearity:
            linearity_placement_list: List[Placement] = [_Partial()]
            for input_dim in input_dims:
                linearity_placement_list.append(_Partial())
            mesh_dim_strategies.append(linearity_placement_list)
        all_mesh_dim_strategies.append(mesh_dim_strategies)
    strategy_combs = itertools.product(*all_mesh_dim_strategies)
    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list = []
        for specs in zip(*strategy_comb):
            spec_list.append(DTensorSpec(mesh, tuple(specs)))
        strat = PlacementStrategy(output_spec=spec_list[0], input_specs=spec_list[1:])
        all_strategies.append(strat)
    return OpStrategy(all_strategies)