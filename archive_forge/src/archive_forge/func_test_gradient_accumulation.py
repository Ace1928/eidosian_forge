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
def test_gradient_accumulation(split_batches=False, dispatch_batches=False, sync_each_batch=False):
    gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=2, sync_each_batch=sync_each_batch)
    accelerator = Accelerator(split_batches=split_batches, dispatch_batches=dispatch_batches, gradient_accumulation_plugin=gradient_accumulation_plugin)
    model, ddp_model, dataloader = get_training_setup(accelerator)
    for iteration, batch in enumerate(dataloader):
        ddp_input, ddp_target = batch.values()
        input, target = accelerator.gather((ddp_input, ddp_target))
        input, target = (input.to(accelerator.device), target.to(accelerator.device))
        step_model(model, input, target, accelerator, False)
        with accelerator.accumulate(ddp_model):
            step_model(ddp_model, ddp_input, ddp_target, accelerator)
        for param, ddp_param in zip(model.parameters(), ddp_model.parameters()):
            if not param.requires_grad:
                continue
            if (iteration + 1) % 2 == 0 or iteration == len(dataloader) - 1 or sync_each_batch:
                assert torch.allclose(param.grad, ddp_param.grad) is True, f'Gradients not in sync when they should be at iteration {iteration}:\nModel grad ({param.grad}) != DDP grad ({ddp_param.grad})'
            else:
                assert torch.allclose(param.grad, ddp_param.grad) is False, f'Gradients in sync when they should not be at iteration {iteration}:\nModel grad ({param.grad}) == DDP grad ({ddp_param.grad})'
        torch.manual_seed(1337 + iteration)
        ddp_input = ddp_input[torch.randperm(len(ddp_input))]
    GradientState._reset_state()