from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from accelerate.accelerator import Accelerator, GradientAccumulationPlugin
from accelerate.state import GradientState
from accelerate.test_utils import RegressionDataset, RegressionModel
from accelerate.utils import DistributedType, set_seed
def test_noop_sync(accelerator):
    model, ddp_model, dataloader = get_training_setup(accelerator)
    ddp_input, ddp_target = next(iter(dataloader)).values()
    for iteration in range(3):
        input, target = accelerator.gather((ddp_input, ddp_target))
        input, target = (input.to(accelerator.device), target.to(accelerator.device))
        step_model(model, input, target, accelerator)
        if iteration % 2 == 0:
            with accelerator.no_sync(ddp_model):
                step_model(ddp_model, ddp_input, ddp_target, accelerator)
        else:
            step_model(ddp_model, ddp_input, ddp_target, accelerator)
        check_model_parameters(model, ddp_model, True, iteration)
        for param, ddp_param in zip(model.parameters(), ddp_model.parameters()):
            if not param.requires_grad:
                continue
            assert torch.allclose(param.grad, ddp_param.grad), f'Gradients not in sync when they should be:\nModel grad ({param.grad}) != DDP grad ({ddp_param.grad})'
        torch.manual_seed(1337 + iteration)
        ddp_input = ddp_input[torch.randperm(len(ddp_input))]