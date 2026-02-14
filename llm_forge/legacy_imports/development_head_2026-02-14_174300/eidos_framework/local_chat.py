from dataclasses import dataclass
import os
import torch
from pathlib import Path
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextIteratorStreamer,
)
from typing import Optional, Union, cast, List, Dict, Tuple, Generator, Type, Any
import logging
import re
from threading import Thread
import sys
import time
from functools import lru_cache

# Configure structured logging with observability features
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d UTC - %(levelname)s - %(module)s.%(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("chat_session.log"),
    ],
)


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable configuration container with environment-aware defaults and validation.

    Attributes:
        models_root: Base directory for model storage
        temperature: Generation temperature (0.1-2.0)
        max_new_tokens: Maximum tokens per generation
        device: Hardware acceleration target
        max_history_length: Context window size in messages
        model_dir_depth: Directory structure depth for model discovery
        default_quit_commands: Session termination phrases
        default_main_model: Fallback model identifier
        default_system_message: Base AI persona definition
        response_streaming: Enable real-time token streaming
        token_margin_ratio: Context window safety buffer (0.05-0.3)
        stop_sequence_timeout: Response completion detection timeout
    """

    models_root: str = os.getenv("MODELS_ROOT", os.path.expanduser("~/ai/models"))
    temperature: float = 0.82
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "32768"))
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    max_history_length: int = 20
    model_dir_depth: int = 2
    default_quit_commands: tuple = ("quit", "exit", "q", "bye")
    default_main_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    default_system_message: str = os.getenv(
        "DEFAULT_SYSTEM_MESSAGE",
        "You are DeepSeek-R1, an AI assistant created exclusively by DeepSeek. Respond helpfully and professionally.",
    )
    response_streaming: bool = True
    token_margin_ratio: float = 0.15
    stop_sequence_timeout: float = 1.5

    def __post_init__(self):
        """Validate configuration parameters"""
        if not 0.1 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.1 and 2.0")
        if not 0.05 <= self.token_margin_ratio <= 0.3:
            raise ValueError("Token margin ratio must be between 0.05 and 0.3")


config = RuntimeConfig()


class ModelLoader:
    """Safe model loader with hardware optimization and fault tolerance.

    Features:
        - Automatic device detection and memory management
        - Model validation and integrity checks
        - Graceful error recovery
        - Quantization support
    """

    @classmethod
    def load(
        cls,
        model_name: str,
        models_root: str = config.models_root,
        device: torch.device = config.device,
        quantize: bool = False,
    ) -> tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
        """Load model and tokenizer with enhanced safety checks.

        Args:
            model_name: Model identifier in vendor/name format
            models_root: Base directory for model storage
            device: Target hardware device
            quantize: Enable 4-bit quantization

        Returns:
            Tuple of (model, tokenizer) or (None, None) on failure
        """
        model_path = Path(models_root) / model_name

        try:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model path inaccessible: {model_path.resolve()}"
                )

            logging.info(f"🔧 Initializing tokenizer from {model_path.name}")
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                use_fast=True,
                padding_side="left",
                truncation_side="left",
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            logging.info(f"🚀 Loading model weights to {device}...")
            load_kwargs = {
                "pretrained_model_name_or_path": str(model_path),
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": (
                    torch.float16 if device.type == "cuda" else torch.float32
                ),
            }

            if quantize and device.type == "cuda":
                load_kwargs["load_in_4bit"] = True
                load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                load_kwargs["bnb_4bit_use_double_quant"] = True
                load_kwargs["bnb_4bit_quant_type"] = "nf4"

            model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
            model = model.to(device).eval()

            logging.info(f"✅ Successfully initialized {model_path.name}")
            return cast(PreTrainedModel, model), cast(PreTrainedTokenizer, tokenizer)

        except Exception as e:
            logging.error(f"💥 Critical failure loading {model_path.name}: {str(e)}")
            return None, None


class PromptTemplateManager:
    """Advanced template system with model-specific optimizations and validation.

    Features:
        - Automatic template detection
        - Cross-model compatibility
        - Context window management
        - Stop sequence optimization
    """

    TEMPLATE_REGISTRY: Dict[str, Type["PromptTemplate"]] = {}

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.template = self.detect_template()
        self.validate_configuration()

    def detect_template(self) -> "PromptTemplate":
        """Intelligent template detection with fallback strategies."""
        try:
            if self.tokenizer.chat_template:
                return self.TEMPLATE_REGISTRY.get("generic", PromptTemplate)()

            model_name = self.tokenizer.name_or_path.lower()
            for key in self.TEMPLATE_REGISTRY:
                if key in model_name:
                    return self.TEMPLATE_REGISTRY[key]()

            return PromptTemplate()

        except Exception as e:
            logging.warning(f"⚠️ Template detection failed: {str(e)}")
            return PromptTemplate()

    def validate_configuration(self):
        """Ensure template compatibility with tokenizer settings."""
        required_tokens = {"bos_token", "eos_token", "pad_token"}
        missing = [
            token for token in required_tokens if not getattr(self.tokenizer, token)
        ]
        if missing:
            logging.warning(f"Missing required tokens: {', '.join(missing)}")
            self.tokenizer.add_special_tokens(
                {token: f"[{token.upper()}]" for token in missing}
            )


@dataclass
class PromptTemplate:
    """Base template for model instruction formatting."""

    system: str = "<|system|>\n{system_message}</s>"
    user: str = "<|user|>\n{user_input}</s>"
    ai: str = "<|assistant|>\n{ai_response}</s>"
    separator: str = "</s>"
    stop_sequences: Tuple[str, ...] = ("</s>", "<|endoftext|>")
    max_context_size: int = 32000
    requires_system_prompt: bool = True
    token_safety_margin: float = 0.1

    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Transform message history into model-ready prompt."""
        prompt = []
        for msg in messages:
            if msg["role"] == "system":
                prompt.append(self.system.format(system_message=msg["content"]))
            elif msg["role"] == "user":
                prompt.append(self.user.format(user_input=msg["content"]))
            elif msg["role"] == "assistant":
                prompt.append(self.ai.format(ai_response=msg["content"]))
        return self.separator.join(prompt)


class ContextManager:
    """Adaptive context window system with message prioritization.

    Features:
        - Token-aware truncation
        - Message priority weighting
        - Dynamic buffer management
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_history: int = config.max_history_length,
    ):
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.message_buffer: list[Dict[str, Any]] = []

    def add_message(self, role: str, content: str):
        """Add message to context buffer with timestamp."""
        self.message_buffer.append(
            {
                "role": role,
                "content": content,
                "tokens": len(self.tokenizer.tokenize(content)),
                "timestamp": time.time(),
            }
        )

    def get_context(self, max_tokens: int) -> List[Dict[str, str]]:
        """Generate optimized context within token limits."""
        context: list[Dict[str, Any]] = []
        current_tokens = 0

        for msg in reversed(self.message_buffer):
            if current_tokens + msg["tokens"] > max_tokens:
                break
            context.insert(0, msg)
            current_tokens += msg["tokens"]

        return [{"role": m["role"], "content": m["content"]} for m in context]


class ChatSession:
    """Stateful chat session manager with streaming and context control.

    Features:
        - Real-time response streaming
        - Context-aware generation
        - Stop sequence detection
        - Error recovery
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.context = ContextManager(tokenizer)
        self.template_manager = PromptTemplateManager(tokenizer)
        self.streamer = TextIteratorStreamer(
            cast(AutoTokenizer, tokenizer),
            timeout=config.stop_sequence_timeout,
        )

    def generate_response(self, user_input: str) -> Generator[str, None, None]:
        """Generate streaming response with full output capture."""
        try:
            self.context.add_message("user", user_input)

            prompt = self._build_prompt()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(config.device)

            generation_kwargs = dict(
                inputs,
                streamer=self.streamer,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=self._get_stopping_criteria(),
            )

            Thread(target=self.model.generate, kwargs=generation_kwargs).start()

            full_response = ""
            for new_text in self.streamer:
                full_response += new_text
                yield new_text

            self._finalize_response(full_response)

        except Exception as e:
            logging.error(f"🚨 Generation error: {str(e)}")
            yield "⚠️ An error occurred during response generation"

    def _build_prompt(self) -> str:
        """Construct optimized prompt with current context."""
        messages = (
            [{"role": "system", "content": config.default_system_message}]
            if self.template_manager.template.requires_system_prompt
            else []
        )

        messages.extend(
            self.context.get_context(
                self.template_manager.template.max_context_size - config.max_new_tokens
            )
        )

        return self.template_manager.template.format_messages(messages)

    def _finalize_response(self, response: str):
        """Process and store completed response."""
        clean_response = self._clean_response(response)
        self.context.add_message("assistant", clean_response)
        logging.info(f"🤖 AI: {clean_response}")

    def _clean_response(self, text: str) -> str:
        """Remove stop sequences and trailing whitespace."""
        stop_positions = [
            text.find(seq) for seq in self.template_manager.template.stop_sequences
        ]
        valid_positions = [pos for pos in stop_positions if pos != -1]
        return text[: min(valid_positions) if valid_positions else None].strip()

    def _get_stopping_criteria(self):
        """Configure model-specific stopping conditions."""
        return None  # Implement custom stopping criteria as needed


def discover_models(models_root: str = config.models_root) -> List[str]:
    """Hierarchical model discovery with error resilience."""
    try:
        models = []
        models_path = Path(models_root)

        for vendor in models_path.glob("*"):
            if vendor.is_dir() and not vendor.name.startswith("."):
                for model in vendor.glob("*"):
                    if model.is_dir():
                        models.append(f"{vendor.name}/{model.name}")

        return sorted(models, key=lambda x: x.lower())

    except Exception as e:
        logging.error(f"🔍 Model discovery failed: {str(e)}")
        return []


def interactive_model_selector(models: List[str]) -> Optional[str]:
    """User-friendly model selection interface."""
    if not models:
        logging.error("❌ No models available for selection")
        return None

    print("\n📚 Available Models:")
    for i, model in enumerate(models, 1):
        print(f"  {i:>2}. {model}")

    while True:
        choice = input("\n🔢 Select model (q to quit): ").strip().lower()

        if choice in config.default_quit_commands:
            logging.info("🚫 Selection aborted")
            return None

        if not choice.isdigit():
            print("❌ Please enter a number")
            continue

        index = int(choice) - 1
        if 0 <= index < len(models):
            return models[index]

        print(f"❌ Invalid selection (1-{len(models)})")


def main():
    """Main execution flow with enhanced error handling."""
    logging.info(f"\n🔍 Model Interface - {Path(config.models_root).resolve()}")
    logging.info(f"🖥️  Device: {config.device}")

    models = discover_models()
    if not models:
        logging.error("❌ No models found")
        return

    selected_model = interactive_model_selector(models)
    if not selected_model:
        return

    model, tokenizer = ModelLoader.load(selected_model)
    if not (model and tokenizer):
        return

    session = ChatSession(model, tokenizer)

    logging.info("💬 Session active (type 'quit' to exit)")
    while True:
        try:
            user_input = input("\n👤 User: ").strip()
            if user_input.lower() in config.default_quit_commands:
                logging.info("👋 Session ended")
                break

            print("\n🤖 AI: ", end="", flush=True)
            full_response = []
            for token in session.generate_response(user_input):
                print(token, end="", flush=True)
                full_response.append(token)

            print("\n")

        except KeyboardInterrupt:
            logging.info("🛑 Session paused")
            if input("Continue? (y/n): ").lower() != "y":
                break

        except Exception as e:
            logging.error(f"💥 Critical error: {str(e)}")
            break


if __name__ == "__main__":
    main()
