# Import PathLike for model paths
from os import PathLike
from typing import Any, Dict, List, Optional, Union, cast

try:  # Optional heavy dependencies
    import torch
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )
except Exception:  # pragma: no cover - optional dependency handling

    class _FakeTorch:
        """Minimal stub when PyTorch is unavailable."""

        Tensor = Any

        @staticmethod
        def no_grad():  # type: ignore
            class _Ctx:
                def __enter__(self):
                    return None

                def __exit__(self, *exc: Any) -> None:
                    pass

            return _Ctx()

    torch = cast(Any, _FakeTorch())
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    PretrainedConfig = Any  # type: ignore
    PreTrainedModel = Any  # type: ignore
    PreTrainedTokenizer = Any  # type: ignore
    PreTrainedTokenizerFast = Any  # type: ignore


class ModelState:
    """Encapsulates a language model and tokenizer instance.

    The previous implementation relied on class-level state and acted as a
    singleton.  This version allows multiple independent model instances to be
    created with different model names or devices.  Lazy initialization is still
    used so the heavy transformer objects are only loaded when required.
    """

    def __init__(
        self,
        model_name: str = "qwen/qwen2.5-0.5b-instruct",
        device: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        if torch is not None and hasattr(torch, "device"):
            self.device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = "cpu"
        self.tokenizer: Optional[
            Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        ] = None
        self.model: Optional[PreTrainedModel] = None
        self._initialized: bool = False
        self._inference_failures: int = 0
        self._max_failures: int = 5
        self._failure_threshold_reached: bool = False

    def get_model_name(self) -> str:
        """Return the configured model name."""
        return self.model_name

    def is_initialized(self) -> bool:
        """Check if the model and tokenizer are initialized."""
        return self._initialized

    def set_model(self, model_name: str) -> None:
        """Change the model and reset initialization state."""
        self.model_name = model_name
        self._initialized = False  # Force reinitialization with new model

    def initialize(self) -> bool:
        """
        Initialize the model and tokenizer if not already done.

        Returns:
            True if initialization was successful, False otherwise
        """
        if self.is_initialized():
            return True

        if self._failure_threshold_reached:
            print(
                f"Model initialization skipped: Failure threshold ({self._max_failures}) reached."
            )
            return False

        try:
            if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
                raise ImportError("transformers or torch not available")

            # Load tokenizer with explicit type annotation
            self.tokenizer = cast(
                Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                AutoTokenizer.from_pretrained(
                    cast(Union[str, PathLike], self.model_name)
                ),
            )
            assert self.tokenizer is not None, "Tokenizer loading returned None"

            # Load model with appropriate configuration
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": (
                    torch.float16
                    if getattr(self.device, "type", "cpu") == "cuda"
                    else torch.float32
                )
            }
            if getattr(self.device, "type", "cpu") == "cuda":
                try:
                    import accelerate  # type: ignore  # noqa: F401

                    model_kwargs["device_map"] = "auto"
                except Exception:
                    model_kwargs["device_map"] = None

            self.model = cast(
                PreTrainedModel,
                AutoModelForCausalLM.from_pretrained(
                    cast(Union[str, PathLike], self.model_name),
                    **model_kwargs,
                ),
            )
            if getattr(self.device, "type", "cpu") == "cpu":
                self.model.to(self.device)
            elif model_kwargs.get("device_map") is None:
                self.model.to(self.device)
            assert self.model is not None, "Model loading returned None"

            self._initialized = True
            print(
                f"Model '{self.model_name}' initialized successfully on {self.device}."
            )
            return True
        except Exception as e:
            print(f"Model initialization failed for '{self.model_name}': {str(e)}")
            self._inference_failures += 1
            if self._inference_failures >= self._max_failures:
                self._failure_threshold_reached = True
                print(
                    f"Failure threshold ({self._max_failures}) reached. Disabling model."
                )
            return False

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = 64,
        temperature: float = 0.7,
        num_beams: int = 3,
    ) -> Optional[str]:
        """
        Generate text using the loaded model.

        Args:
            prompt: Input text to generate from
            max_new_tokens: Maximum number of tokens to generate (None for model's maximum capacity)
            temperature: Sampling temperature
            num_beams: Number of beams for beam search

        Returns:
            Generated text or None if generation failed
        """
        if not self.initialize():
            return None

        if torch is None:
            print("Text generation skipped: torch not available")
            return None

        # Safety check - both must be initialized
        if self.tokenizer is None or self.model is None:
            print("Error: Tokenizer or model is None after initialization attempt.")
            return None

        try:
            # Create input tensors
            input_tokens: Dict[str, torch.Tensor] = self.tokenizer(
                prompt, return_tensors="pt"
            )  # type: ignore

            # Safely access 'input_ids' and 'attention_mask'
            input_ids_tensor = input_tokens.get("input_ids")
            if input_ids_tensor is None:
                raise ValueError("Tokenizer did not return 'input_ids'")
            if not isinstance(input_ids_tensor, torch.Tensor):
                raise TypeError(
                    f"Expected input_ids to be a Tensor, got {type(input_ids_tensor)}"
                )
            input_ids = input_ids_tensor.to(self.device)

            attention_mask_tensor = input_tokens.get("attention_mask")
            attention_mask = None
            if attention_mask_tensor is not None:
                if not isinstance(attention_mask_tensor, torch.Tensor):
                    raise TypeError(
                        f"Expected attention_mask to be a Tensor, got {type(attention_mask_tensor)}"
                    )
                attention_mask = attention_mask_tensor.to(self.device)

            # Configure generation parameters
            gen_kwargs: Dict[str, Any] = {
                "temperature": temperature,
                "num_beams": num_beams,
                "do_sample": temperature > 0,
            }

            # Calculate max_length carefully
            input_length = input_ids.shape[1] if hasattr(input_ids, "shape") else 0
            model_max_length = 2048
            model_config: Optional[PretrainedConfig] = getattr(
                self.model, "config", None
            )
            if model_config and hasattr(model_config, "max_position_embeddings"):
                model_max_length = getattr(
                    model_config, "max_position_embeddings", model_max_length
                )

            if max_new_tokens is None:
                gen_kwargs["max_length"] = model_max_length
            else:
                gen_kwargs["max_length"] = min(
                    input_length + max_new_tokens, model_max_length
                )

            # Handle pad_token_id carefully
            pad_token_id: Optional[Union[int, List[int]]] = self.tokenizer.pad_token_id  # type: ignore
            eos_token_id: Optional[Union[int, List[int]]] = self.tokenizer.eos_token_id  # type: ignore

            if pad_token_id is None:
                if eos_token_id is not None:
                    gen_kwargs["pad_token_id"] = (
                        eos_token_id[0]
                        if isinstance(eos_token_id, list)
                        else eos_token_id
                    )
                else:
                    print(
                        "Warning: Tokenizer lacks both pad_token_id and eos_token_id. Generation might be unstable."
                    )
            else:
                gen_kwargs["pad_token_id"] = (
                    pad_token_id[0] if isinstance(pad_token_id, list) else pad_token_id
                )

            if eos_token_id is not None:
                gen_kwargs["eos_token_id"] = (
                    eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
                )

            # Generate text
            with torch.no_grad():
                outputs: torch.Tensor = self.model.generate(  # type: ignore
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

            # Process the output
            output_sequence: Optional[torch.Tensor] = None
            if isinstance(outputs, torch.Tensor):
                output_sequence = outputs
            elif hasattr(outputs, "sequences"):
                output_sequence = getattr(outputs, "sequences", None)

            if output_sequence is None or not isinstance(output_sequence, torch.Tensor):
                print(
                    f"Warning: Could not extract output sequences from model.generate output type: {type(outputs)}"
                )
                newly_generated_ids = torch.tensor(
                    [], dtype=torch.long, device=self.device
                )
            else:
                first_sequence = (
                    output_sequence[0]
                    if output_sequence.ndim > 1 and output_sequence.shape[0] > 0
                    else output_sequence
                )

                if first_sequence.shape[0] > input_length:
                    newly_generated_ids = first_sequence[input_length:]
                else:
                    newly_generated_ids = torch.tensor(
                        [], dtype=torch.long, device=self.device
                    )

            result = self.tokenizer.decode(newly_generated_ids, skip_special_tokens=True)  # type: ignore

            return result.strip()

        except Exception as e:
            self._inference_failures += 1
            if self._inference_failures >= self._max_failures:
                self._failure_threshold_reached = True
                print(
                    f"Failure threshold ({self._max_failures}) reached. Disabling model."
                )
            print(f"Text generation failed: {str(e)}")
            return None

    def query(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = 256,
        temperature: float = 0.7,
        num_beams: int = 3,
    ) -> Optional[str]:
        """
        Query the model with a prompt.

        Args:
            prompt: Input text to query
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            num_beams: Number of beams for beam search

        Returns:
            Generated text or None if generation failed
        """
        return self.generate_text(prompt, max_new_tokens, temperature, num_beams)
