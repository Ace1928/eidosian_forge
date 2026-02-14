from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore
from accelerate import Accelerator  # type: ignore


def update_qwen_model_with_dynamic_tokenizer(
    dynamic_tokenizer, model_name="Qwen/Qwen2.5-0.5b"
):
    """
    Load the Qwen/Qwen2.5 model from the pretrained repository, update its configuration
    with the dynamic tokenizer's vocabulary size, and resize its embedding matrix.
    The model is then prepared for CPU-only operation using the Accelerate library with disk offload.
    """
    # Load the original configuration and update the vocabulary size.
    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size = len(dynamic_tokenizer.vocab)

    # Load the pretrained model with a flag to ignore mismatched embedding sizes.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True
    )

    # Resize token embeddings to include newly learned tokens.
    model.resize_token_embeddings(config.vocab_size)

    # Prepare the model for CPU-only execution (with disk offload if necessary).
    accelerator = Accelerator(cpu=True)
    model = accelerator.prepare(model)

    return model
