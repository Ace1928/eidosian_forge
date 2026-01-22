from typing import List
import torch
def get_all_qint_dtypes() -> List[torch.dtype]:
    return [torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4]