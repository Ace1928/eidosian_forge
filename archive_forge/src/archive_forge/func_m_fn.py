import torch
def m_fn(m, d):
    if isinstance(m, torch.nn.Linear):
        return MkldnnLinear(m, d)
    elif isinstance(m, torch.nn.Conv1d):
        return MkldnnConv1d(m, d)
    elif isinstance(m, torch.nn.Conv2d):
        return MkldnnConv2d(m, d)
    elif isinstance(m, torch.nn.Conv3d):
        return MkldnnConv3d(m, d)
    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        return MkldnnBatchNorm(m)
    elif isinstance(m, torch.nn.PReLU):
        return MkldnnPrelu(m, d)
    else:
        return m