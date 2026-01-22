from typing import Iterator, List, Tuple
def leaderboard() -> Iterator[Tuple[float, int]]:
    return ((block_size(i), i) for i in range(partitions))