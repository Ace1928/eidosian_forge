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
def test_gradient_accumulation_with_opt_and_scheduler(split_batches=False, dispatch_batches=False, sync_each_batch=False):
    gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=2, sync_each_batch=sync_each_batch)
    accelerator = Accelerator(split_batches=split_batches, dispatch_batches=dispatch_batches, gradient_accumulation_plugin=gradient_accumulation_plugin)
    model, opt, sched, dataloader, ddp_model, ddp_opt, ddp_sched = get_training_setup(accelerator, True)
    for iteration, batch in enumerate(dataloader):
        ddp_input, ddp_target = batch.values()
        input, target = accelerator.gather((ddp_input, ddp_target))
        input, target = (input.to(accelerator.device), target.to(accelerator.device))
        model.train()
        ddp_model.train()
        step_model(model, input, target, accelerator, False)
        opt.step()
        if (iteration + 1) % 2 == 0 or iteration + 1 == len(dataloader) or sync_each_batch:
            if split_batches:
                sched.step()
            else:
                for _ in range(accelerator.num_processes):
                    sched.step()
        with accelerator.accumulate(ddp_model):
            step_model(ddp_model, ddp_input, ddp_target, accelerator)
            ddp_opt.step()
            ddp_sched.step()
        assert opt.param_groups[0]['lr'] == ddp_opt.param_groups[0]['lr'], f'Learning rates found in each optimizer did not align\nopt: {opt.param_groups[0]['lr']}\nDDP opt: {ddp_opt.param_groups[0]['lr']}\n'
        did_step = (iteration + 1) % 2 == 0 or iteration + 1 == len(dataloader) or sync_each_batch
        if accelerator.num_processes > 1:
            check_model_parameters(model, ddp_model, did_step, iteration, rtol=0.001)
        if (iteration + 1) % 2 == 0 or iteration + 1 == len(dataloader) or sync_each_batch:
            opt.zero_grad()
        ddp_opt.zero_grad()
        torch.manual_seed(1337 + iteration)
    GradientState._reset_state()