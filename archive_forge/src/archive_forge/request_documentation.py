from dataclasses import dataclass

    Request for a LoRA adapter.

    Note that this class should be be used internally. For online
    serving, it is recommended to not allow users to use this class but
    instead provide another layer of abstraction to prevent users from
    accessing unauthorized LoRA adapters.

    lora_int_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.
    