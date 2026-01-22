import itertools
from typing import List, Optional, Set, Tuple, cast
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
@triton.jit
def determine_tile(A, B1, B2, B3, C1, C2, C3, A_my_shard, C1_my_shard, C2_my_shard, C3_my_shard, M, N1, N2, N3, my_rank, world_size, direction, stride_am, stride_bk1, stride_bk2, stride_bk3, stride_bn1, stride_bn2, stride_bn3, stride_cm1, stride_cm2, stride_cm3, stride_cn1, stride_cn2, stride_cn3, BLOCK_M, BLOCK_N, GROUP_M):
    M_per_rank = M // world_size
    pid = tl.program_id(0)
    grid_m_per_rank = tl.cdiv(M_per_rank, BLOCK_M)
    grid_n1 = tl.cdiv(N1, BLOCK_N)
    grid_n2 = tl.cdiv(N2, BLOCK_N)
    grid_n3 = tl.cdiv(N3, BLOCK_N)
    grid_n = grid_n1 + grid_n2 + grid_n3
    blocks_per_rank = grid_m_per_rank * grid_n
    if direction == BACKWARDS_WITH_ME_FIRST:
        other_rank = (my_rank - pid // blocks_per_rank + world_size) % world_size
    else:
        other_rank = (my_rank + (pid // blocks_per_rank + 1)) % world_size
    pid = pid % blocks_per_rank
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m_per_rank - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    B = tl.where(pid_n < grid_n1, B1, tl.where(pid_n < grid_n1 + grid_n2, B2, B3))
    C = tl.where(pid_n < grid_n1, C1, tl.where(pid_n < grid_n1 + grid_n2, C2, C3))
    C_my_shard = tl.where(pid_n < grid_n1, C1_my_shard, tl.where(pid_n < grid_n1 + grid_n2, C2_my_shard, C3_my_shard))
    stride_bk = tl.where(pid_n < grid_n1, stride_bk1, tl.where(pid_n < grid_n1 + grid_n2, stride_bk2, stride_bk3))
    stride_bn = tl.where(pid_n < grid_n1, stride_bn1, tl.where(pid_n < grid_n1 + grid_n2, stride_bn2, stride_bn3))
    stride_cm = tl.where(pid_n < grid_n1, stride_cm1, tl.where(pid_n < grid_n1 + grid_n2, stride_cm2, stride_cm3))
    stride_cn = tl.where(pid_n < grid_n1, stride_cn1, tl.where(pid_n < grid_n1 + grid_n2, stride_cn2, stride_cn3))
    N = tl.where(pid_n < grid_n1, N1, tl.where(pid_n < grid_n1 + grid_n2, N2, N3))
    pid_n = tl.where(pid_n < grid_n1, pid_n, tl.where(pid_n < grid_n1 + grid_n2, pid_n - grid_n1, pid_n - grid_n1 - grid_n2))
    A = tl.where(other_rank == my_rank, A_my_shard, A + other_rank * M_per_rank * stride_am)
    C = tl.where(other_rank == my_rank, C_my_shard, C + other_rank * M_per_rank * stride_cm)
    return (A, B, C, M_per_rank, N, stride_bk, stride_bn, stride_cm, stride_cn, pid_m, pid_n, other_rank, blocks_per_rank)