def log_output_artifact(name, data, type, run=None):
    artifact = wandb.Artifact(name, type=type)
    artifact.add_file(data)
    run.log_artifact(artifact)
    wandb.termlog(f'Logging artifact: {name}')