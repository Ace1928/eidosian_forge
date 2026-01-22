import os
import torch
from ..logging import get_logger
from .constants import FSDP_MODEL_NAME, FSDP_PYTORCH_VERSION, OPTIMIZER_NAME
from .imports import is_torch_distributed_available
from .modeling import is_peft_model
from .versions import is_torch_version
def save_fsdp_optimizer(fsdp_plugin, accelerator, optimizer, model, output_dir, optimizer_index=0):
    os.makedirs(output_dir, exist_ok=True)
    with FSDP.state_dict_type(model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config):
        optim_state = FSDP.optim_state_dict(model, optimizer)
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            if accelerator.process_index == 0:
                optim_state_name = f'{OPTIMIZER_NAME}.bin' if optimizer_index == 0 else f'{OPTIMIZER_NAME}_{optimizer_index}.bin'
                output_optimizer_file = os.path.join(output_dir, optim_state_name)
                logger.info(f'Saving Optimizer state to {output_optimizer_file}')
                torch.save(optim_state, output_optimizer_file)
                logger.info(f'Optimizer state saved in {output_optimizer_file}')
        else:
            ckpt_dir = os.path.join(output_dir, f'{OPTIMIZER_NAME}_{optimizer_index}')
            os.makedirs(ckpt_dir, exist_ok=True)
            logger.info(f'Saving Optimizer state to {ckpt_dir}')
            dist_cp.save_state_dict(state_dict={'optimizer': optim_state}, storage_writer=dist_cp.FileSystemWriter(ckpt_dir), planner=DefaultSavePlanner())
            logger.info(f'Optimizer state saved in {ckpt_dir}')