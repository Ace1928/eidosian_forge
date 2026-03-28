# Import PathLike for model paths
import os
import re
import shutil
import subprocess
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, cast

from eidosian_core import eidosian
from eidosian_core.ports import get_service_url

OLLAMA_GENERATE_URL = get_service_url(
    "ollama_http", default_port=11434, default_host="localhost", default_path="/api/generate"
)

FORGE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REMOTE_MODEL = "qwen/qwen3.5-2b-instruct"
DEFAULT_LOCAL_GGUF_CANDIDATES = (
    FORGE_ROOT / "models" / "Qwen3.5-2B-IQ4_XS.gguf",
    FORGE_ROOT / "models" / "Qwen3.5-2B-Q4_K_M.gguf",
    FORGE_ROOT / "models" / "Qwen3.5-0.8B-Q4_K_M.gguf",
    FORGE_ROOT / "models" / "Qwen3.5-0.8B-IQ4_XS.gguf",
    FORGE_ROOT / "models" / "Qwen2.5-0.5B-Instruct-Q8_0.gguf",
)
DEFAULT_LLAMA_CLI_CANDIDATES = (
    FORGE_ROOT / "llama.cpp" / "build" / "bin" / "llama-cli",
    FORGE_ROOT / "llm_forge" / "bin" / "llama-cli",
)


def _resolve_existing_path(candidates: Sequence[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_llama_cli_path() -> Optional[Path]:
    env_path = os.environ.get("EIDOS_WORD_FORGE_LLAMA_CLI") or os.environ.get("EIDOS_LLAMA_CPP_CLI_PATH")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path
    direct = _resolve_existing_path(DEFAULT_LLAMA_CLI_CANDIDATES)
    if direct is not None:
        return direct
    which_path = shutil.which("llama-cli")
    return Path(which_path) if which_path else None


def default_word_forge_model_name() -> str:
    override = os.environ.get("EIDOS_WORD_FORGE_LLM_MODEL")
    if override:
        return override
    explicit_local = os.environ.get("EIDOS_WORD_FORGE_GGUF_MODEL")
    if explicit_local:
        path = Path(explicit_local).expanduser()
        if path.exists():
            return f"gguf:{path}"
    local_model = _resolve_existing_path(DEFAULT_LOCAL_GGUF_CANDIDATES)
    if local_model is not None:
        return f"gguf:{local_model}"
    return DEFAULT_REMOTE_MODEL


def _coerce_gguf_model_path(model_name: str) -> Optional[Path]:
    candidate = model_name.strip()
    if candidate.startswith("gguf:"):
        candidate = candidate.split("gguf:", 1)[1].strip()
    if not candidate:
        return None
    if candidate.endswith(".gguf"):
        path = Path(candidate).expanduser()
        return path if path.exists() else None
    return None


def _clean_llama_cli_output(raw_output: str) -> str:
    text = raw_output.replace("\r\n", "\n")
    text = text.replace("Exiting...", "")
    if "\n\n> " in text:
        text = text.split("\n\n> ", 1)[1]
        blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
        if len(blocks) >= 2:
            text = blocks[1]
        elif blocks:
            text = blocks[-1]
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

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
        model_name: Optional[str] = None,
        device: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name or default_word_forge_model_name()
        self._ollama_model: Optional[str] = None
        self._gguf_model_path: Optional[Path] = None
        self._llama_cli_path: Optional[Path] = None
        if torch is not None and hasattr(torch, "device"):
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None
        self.model: Optional[PreTrainedModel] = None
        self._initialized: bool = False
        self._inference_failures: int = 0
        self._max_failures: int = 5
        self._failure_threshold_reached: bool = False

    @eidosian()
    def get_model_name(self) -> str:
        """Return the configured model name."""
        return self.model_name

    @eidosian()
    def is_initialized(self) -> bool:
        """Check if the model and tokenizer are initialized."""
        return self._initialized

    @eidosian()
    def set_model(self, model_name: str) -> None:
        """Change the model and reset initialization state."""
        self.model_name = model_name
        self._ollama_model = None
        self._gguf_model_path = None
        self._llama_cli_path = None
        self._initialized = False  # Force reinitialization with new model

    @eidosian()
    def initialize(self) -> bool:
        """
        Initialize the model and tokenizer if not already done.

        Returns:
            True if initialization was successful, False otherwise
        """
        if self.is_initialized():
            return True

        if self._failure_threshold_reached:
            print(f"Model initialization skipped: Failure threshold ({self._max_failures}) reached.")
            return False

        try:
            if self.model_name.startswith("ollama:"):
                self._ollama_model = self.model_name.split("ollama:", 1)[1].strip()
                if not self._ollama_model:
                    raise ValueError("Ollama model name is empty")
                self._initialized = True
                return True

            gguf_model_path = _coerce_gguf_model_path(self.model_name)
            if gguf_model_path is not None:
                llama_cli_path = _resolve_llama_cli_path()
                if llama_cli_path is None:
                    raise FileNotFoundError("llama-cli not found for GGUF model backend")
                self._gguf_model_path = gguf_model_path
                self._llama_cli_path = llama_cli_path
                self._initialized = True
                return True

            if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
                raise ImportError("transformers or torch not available")

            # Load tokenizer with explicit type annotation
            self.tokenizer = cast(
                Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                AutoTokenizer.from_pretrained(cast(Union[str, PathLike], self.model_name)),
            )
            assert self.tokenizer is not None, "Tokenizer loading returned None"

            # Load model with appropriate configuration
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": (torch.float16 if getattr(self.device, "type", "cpu") == "cuda" else torch.float32)
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
            print(f"Model '{self.model_name}' initialized successfully on {self.device}.")
            return True
        except Exception as e:
            print(f"Model initialization failed for '{self.model_name}': {str(e)}")
            self._inference_failures += 1
            if self._inference_failures >= self._max_failures:
                self._failure_threshold_reached = True
                print(f"Failure threshold ({self._max_failures}) reached. Disabling model.")
            return False

    @eidosian()
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

        if self.model_name.startswith("ollama:"):
            return self._generate_text_ollama(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        if self._gguf_model_path is not None:
            return self._generate_text_gguf(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        if torch is None:
            print("Text generation skipped: torch not available")
            return None

        if self.tokenizer is None or self.model is None:
            print("Error: Tokenizer or model is None after initialization attempt.")
            return None

        try:
            input_tokens: Dict[str, torch.Tensor] = self.tokenizer(prompt, return_tensors="pt")  # type: ignore

            input_ids_tensor = input_tokens.get("input_ids")
            if input_ids_tensor is None:
                raise ValueError("Tokenizer did not return 'input_ids'")
            if not isinstance(input_ids_tensor, torch.Tensor):
                raise TypeError(f"Expected input_ids to be a Tensor, got {type(input_ids_tensor)}")
            input_ids = input_ids_tensor.to(self.device)

            attention_mask_tensor = input_tokens.get("attention_mask")
            attention_mask = None
            if attention_mask_tensor is not None:
                if not isinstance(attention_mask_tensor, torch.Tensor):
                    raise TypeError(f"Expected attention_mask to be a Tensor, got {type(attention_mask_tensor)}")
                attention_mask = attention_mask_tensor.to(self.device)

            gen_kwargs: Dict[str, Any] = {
                "temperature": temperature,
                "num_beams": num_beams,
                "do_sample": temperature > 0,
            }

            input_length = input_ids.shape[1] if hasattr(input_ids, "shape") else 0
            model_max_length = 2048
            model_config: Optional[PretrainedConfig] = getattr(self.model, "config", None)
            if model_config and hasattr(model_config, "max_position_embeddings"):
                model_max_length = getattr(model_config, "max_position_embeddings", model_max_length)

            if max_new_tokens is None:
                gen_kwargs["max_length"] = model_max_length
            else:
                gen_kwargs["max_length"] = min(input_length + max_new_tokens, model_max_length)

            pad_token_id: Optional[Union[int, List[int]]] = self.tokenizer.pad_token_id  # type: ignore
            eos_token_id: Optional[Union[int, List[int]]] = self.tokenizer.eos_token_id  # type: ignore

            if pad_token_id is None:
                if eos_token_id is not None:
                    gen_kwargs["pad_token_id"] = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
                else:
                    print("Warning: Tokenizer lacks both pad_token_id and eos_token_id. Generation might be unstable.")
            else:
                gen_kwargs["pad_token_id"] = pad_token_id[0] if isinstance(pad_token_id, list) else pad_token_id

            if eos_token_id is not None:
                gen_kwargs["eos_token_id"] = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id

            with torch.no_grad():
                outputs: torch.Tensor = self.model.generate(  # type: ignore
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

            output_sequence: Optional[torch.Tensor] = None
            if isinstance(outputs, torch.Tensor):
                output_sequence = outputs
            elif hasattr(outputs, "sequences"):
                output_sequence = getattr(outputs, "sequences", None)

            if output_sequence is None or not isinstance(output_sequence, torch.Tensor):
                print(f"Warning: Could not extract output sequences from model.generate output type: {type(outputs)}")
                newly_generated_ids = torch.tensor([], dtype=torch.long, device=self.device)
            else:
                first_sequence = (
                    output_sequence[0] if output_sequence.ndim > 1 and output_sequence.shape[0] > 0 else output_sequence
                )
                if first_sequence.shape[0] > input_length:
                    newly_generated_ids = first_sequence[input_length:]
                else:
                    newly_generated_ids = torch.tensor([], dtype=torch.long, device=self.device)

            result = self.tokenizer.decode(newly_generated_ids, skip_special_tokens=True)  # type: ignore
            return result.strip()

        except Exception as e:
            self._inference_failures += 1
            if self._inference_failures >= self._max_failures:
                self._failure_threshold_reached = True
                print(f"Failure threshold ({self._max_failures}) reached. Disabling model.")
            print(f"Text generation failed: {str(e)}")
            return None

    def _generate_text_ollama(
        self,
        prompt: str,
        max_new_tokens: Optional[int],
        temperature: float,
    ) -> Optional[str]:
        """Generate text via local Ollama server."""
        if not self._ollama_model:
            print("Ollama model not configured.")
            return None

        try:
            import requests  # type: ignore

            if max_new_tokens is None:
                max_new_tokens = 64
            if max_new_tokens > 128:
                max_new_tokens = 128

            payload: Dict[str, Any] = {
                "model": self._ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            }
            if max_new_tokens is not None:
                payload["options"]["num_predict"] = max_new_tokens

            resp = requests.post(
                OLLAMA_GENERATE_URL,
                json=payload,
                timeout=900,
            )
            resp.raise_for_status()
            data = resp.json()
            response = data.get("response")
            if not isinstance(response, str):
                return None
            return response.strip()
        except Exception as e:
            self._inference_failures += 1
            if self._inference_failures >= self._max_failures:
                self._failure_threshold_reached = True
                print(f"Failure threshold ({self._max_failures}) reached. Disabling model.")
            print(f"Ollama generation failed: {str(e)}")
            return None

    def _generate_text_gguf(
        self,
        prompt: str,
        max_new_tokens: Optional[int],
        temperature: float,
    ) -> Optional[str]:
        """Generate text via local llama.cpp GGUF inference."""
        if self._gguf_model_path is None or self._llama_cli_path is None:
            print("GGUF model or llama-cli path is not configured.")
            return None

        try:
            if max_new_tokens is None:
                max_new_tokens = 64
            if max_new_tokens > 128:
                max_new_tokens = 128

            cmd = [
                str(self._llama_cli_path),
                "-m",
                str(self._gguf_model_path),
                "-p",
                prompt,
                "-st",
                "--reasoning-budget",
                "0",
                "-n",
                str(max_new_tokens),
                "-c",
                "1024",
                "-t",
                str(min(max((os.cpu_count() or 2) // 2, 1), 4)),
                "--temp",
                str(temperature),
                "--top-k",
                "20",
                "--top-p",
                "0.9",
                "--simple-io",
                "--no-display-prompt",
                "--no-show-timings",
                "--log-disable",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                env=self._subprocess_env(),
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"llama-cli exited {result.returncode}")
            response = _clean_llama_cli_output(result.stdout)
            return response or None
        except Exception as e:
            self._inference_failures += 1
            if self._inference_failures >= self._max_failures:
                self._failure_threshold_reached = True
                print(f"Failure threshold ({self._max_failures}) reached. Disabling model.")
            print(f"GGUF generation failed: {str(e)}")
            return None

    def _subprocess_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        if self._llama_cli_path is None:
            return env
        bin_dir = str(self._llama_cli_path.parent.resolve())
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        ld_library = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{ld_library}" if ld_library else bin_dir
        return env

    @eidosian()
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
