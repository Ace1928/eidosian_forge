def get_model_with_peft_adapter(base_model, peft_adapter_path):
    """
    Apply the PEFT adapter to the base model to create a PEFT model.

    NB: The alternative way to load PEFT adapter is to use load_adapter API like
    `base_model.load_adapter(peft_adapter_path)`, as it injects the adapter weights
    into the model in-place hence reducing the memory footprint. However, doing so
    returns the base model class and not the PEFT model, loosing some properties
    such as peft_config. This is not preferable because load_model API should
    return the exact same object that was saved. Hence we construct the PEFT model
    instead of in-place injection, for consistency over the memory saving which
    should be small in most cases.
    """
    from peft import PeftModel
    return PeftModel.from_pretrained(base_model, peft_adapter_path)