def get_peft_base_model(model):
    """Extract the base model from a PEFT model."""
    peft_config = model.peft_config.get(model.active_adapter) if model.peft_config else None
    if peft_config and (not peft_config.is_prompt_learning):
        return model.base_model.model
    return model.base_model