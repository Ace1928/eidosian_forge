import inspect
import pickle
from functools import wraps
from pathlib import Path
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
@typedispatch
def wandb_track(name: str, data: BaseEstimator, models=False, run=None, testing=False, *args, **kwargs):
    if testing:
        return 'BaseEstimator' if models else None
    if models:
        artifact = wandb.Artifact(name, type='model')
        with artifact.new_file(f'{name}.pkl', 'wb') as f:
            pickle.dump(data, f)
        run.log_artifact(artifact)
        wandb.termlog(f'Logging artifact: {name} ({type(data)})')