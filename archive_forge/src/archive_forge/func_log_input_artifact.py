def log_input_artifact(name, data, type, run=None):
    artifact = wandb.Artifact(name, type=type)
    artifact.add_file(data)
    run.use_artifact(artifact)
    wandb.termlog(f'Using artifact: {name}')