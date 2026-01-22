import wandb
from . import preinit
def unset_globals():
    wandb.run = None
    wandb.config = preinit.PreInitObject('wandb.config')
    wandb.summary = preinit.PreInitObject('wandb.summary')
    wandb.log = preinit.PreInitCallable('wandb.log', wandb.wandb_sdk.wandb_run.Run.log)
    wandb.save = preinit.PreInitCallable('wandb.save', wandb.wandb_sdk.wandb_run.Run.save)
    wandb.use_artifact = preinit.PreInitCallable('wandb.use_artifact', wandb.wandb_sdk.wandb_run.Run.use_artifact)
    wandb.log_artifact = preinit.PreInitCallable('wandb.log_artifact', wandb.wandb_sdk.wandb_run.Run.log_artifact)
    wandb.define_metric = preinit.PreInitCallable('wandb.define_metric', wandb.wandb_sdk.wandb_run.Run.define_metric)