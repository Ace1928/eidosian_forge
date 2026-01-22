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
def test_distributed_sync_multiple_fwd(accelerator):
    model, ddp_model, dataloader = get_training_setup(accelerator)
    losses = []
    num_iterations = 3
    for iteration in range(num_iterations):
        ddp_input, ddp_target = next(iter(dataloader)).values()
        input, target = accelerator.gather((ddp_input, ddp_target))
        input, target = (input.to(accelerator.device), target.to(accelerator.device))
        step_model(model, input, target, accelerator)
        with accelerator.no_sync(ddp_model):
            ddp_output = ddp_model(ddp_input)
            loss = F.mse_loss(ddp_output, ddp_target.to(ddp_output.device))
            losses.append(loss)
    for iteration in range(num_iterations):
        loss = losses[iteration]
        if iteration < num_iterations - 1:
            accelerator.backward(loss)
            for param, ddp_param in zip(model.parameters(), ddp_model.parameters()):
                if not param.requires_grad:
                    continue
                assert torch.allclose(param.grad, ddp_param.grad) is False, f'Gradients in sync when they should not be:\nModel grad ({param.grad}) == DDP grad ({ddp_param.grad})'
        else:
            with accelerator.trigger_sync_in_backward(ddp_model):
                accelerator.backward(loss)
            for param, ddp_param in zip(model.parameters(), ddp_model.parameters()):
                if not param.requires_grad:
                    continue
                assert torch.allclose(param.grad, ddp_param.grad) is True, f'Gradients not in sync when they should be:\nModel grad ({param.grad}) != DDP grad ({ddp_param.grad})'