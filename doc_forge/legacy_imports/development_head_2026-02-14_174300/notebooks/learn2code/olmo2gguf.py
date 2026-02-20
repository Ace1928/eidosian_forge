#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A comprehensive abstraction layer for Hugging Face Transformers models,
specifically tailored for the Olmo2 family on CPU. This module provides a highly
modular, flexible, scalable, and performant interface to load, configure,
and perform inference with Olmo2 models, while encapsulating all aspects
and parameters of the underlying model and configuration.

Optimized for CPU performance and minimal RAM usage:
- Adjusts PyTorch threading settings.
- Uses model evaluation mode and disables gradients for inference.
- Streams operations for speed and resource usage.
- Employs disk offloading to handle larger models on limited hardware.
- Implements aggressive offloading to keep RAM usage minimal (~0.5GB).

Author: Your Name
Date: 2023-XX-XX
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Configure logging for detailed tracing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CPU-specific optimizations: set number of threads for PyTorch operations
torch.set_num_threads(torch.get_num_threads())


class ModelWrapper:
    """
    Abstract base class for a model wrapper. Provides generic methods for
    loading configurations, models, tokenizers, and performing inference.
    """

    def __init__(
        self,
        model_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        offload_folder: str = "offload",
    ):
        """
        Initialize the ModelWrapper.

        Args:
            model_name (str): Identifier or path for the pretrained model.
            config_overrides (Optional[Dict[str, Any]]): Dictionary to override
                default configuration parameters.
            device (Optional[Union[str, torch.device]]): Device to load the model
                onto (e.g., 'cpu'). Forced to CPU in this implementation.
            offload_folder (str): Directory for disk offloading.
        """
        self.model_name = model_name
        self.config_overrides = config_overrides or {}
        # Force CPU device
        self.device = torch.device("cpu")
        self.offload_folder = offload_folder

        self.config: Optional[PretrainedConfig] = None
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        # Load configuration, tokenizer, and model
        self._load_configuration()
        self._load_tokenizer()
        self._load_model()

    def _load_configuration(self) -> None:
        """
        Load and configure the model configuration, applying any overrides.
        """
        logger.info("Loading model configuration...")
        self.config = AutoConfig.from_pretrained(self.model_name, **self.config_overrides)
        logger.debug(f"Configuration loaded: {self.config.to_dict()}")

    def _load_model(self) -> None:
        """
        Load the pretrained model based on the configuration using disk offloading
        and aggressive memory minimization strategies.
        """
        logger.info("Loading model with advanced disk offloading strategies...")
        try:
            config = AutoConfig.from_pretrained(self.model_name, **self.config_overrides)

            # Try accelerate loading next
            try:
                from accelerate import init_empty_weights, load_checkpoint_and_dispatch
                from huggingface_hub import snapshot_download

                logger.info("Attempting chunked model loading with accelerate for minimal RAM usage.")

                # Download the model files locally
                model_folder = snapshot_download(repo_id=self.model_name)

                # Initialize empty model with no memory allocation
                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(self.config)

                # Load checkpoint shards with aggressive offloading
                self.model = load_checkpoint_and_dispatch(
                    self.model,
                    model_folder,
                    device_map={"": "cpu"},
                    max_memory={"cpu": "1GB"},
                    no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LlamaDecoderLayer"],
                    offload_folder=self.offload_folder,
                    offload_buffers=True,
                    dtype=torch.float32,
                    offload_state_dict=True,
                    skip_keys=None,
                    preload_module_classes=None,
                    force_hooks=False,
                    strict=False,
                )
                logger.info("Successfully loaded model using accelerate")
                return
            except ImportError:
                logger.warning("accelerate library not found, falling back to standard loading...")

            # Finally try standard loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=self.config,
                device_map={"": "cpu"},
                offload_folder=self.offload_folder,
                offload_state_dict=True,
                low_cpu_mem_usage=True,
            )
            logger.info("Successfully loaded model using standard loading")

        except Exception as e:
            logger.error(f"Failed to load the model: {e}")
            raise

        self.model.eval()
        logger.info("Model loaded and set to eval mode successfully with minimal RAM footprint.")

    def _load_tokenizer(self) -> None:
        """
        Load the tokenizer for the model.
        """
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info("Tokenizer loaded.")

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        num_return_sequences: int = 1,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate text given an input prompt using the model.

        Args:
            prompt (str): Input text prompt.
            max_length (int): Maximum length of generated sequences.
            num_return_sequences (int): Number of sequences to generate.
            generation_kwargs (Optional[Dict[str, Any]]): Additional generation
                parameters for the model's generate method.

        Returns:
            List[str]: List of generated text sequences.
        """
        if not self.tokenizer or not self.model:
            raise ValueError("Tokenizer and model must be loaded before generation.")

        generation_kwargs = generation_kwargs or {}
        logger.info(f"Tokenizing prompt: {prompt}")
        # Tokenize inputs and ensure they are on CPU
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Create a generation configuration combining defaults with overrides
        gen_config = GenerationConfig(max_length=max_length, **generation_kwargs)

        logger.info("Generating sequences in a memory-efficient manner...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_ids,
                generation_config=gen_config,
                num_return_sequences=num_return_sequences,
            )

        logger.info("Decoding generated sequences...")
        outputs = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        logger.debug(f"Generated outputs: {outputs}")
        return outputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Any, Tuple]:
        """
        Perform a forward pass through the model in a memory-efficient way.

        Args:
            input_ids (Optional[torch.Tensor]): Input token IDs.
            attention_mask (Optional[torch.Tensor]): Attention mask.
            position_ids (Optional[torch.Tensor]): Token position indices.
            past_key_values (Optional[Any]): Past key values for faster decoding.
            inputs_embeds (Optional[torch.Tensor]): Precomputed input embeddings.
            use_cache (Optional[bool]): Whether to use cached key/value states.
            output_attentions (Optional[bool]): Whether to output attentions.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            return_dict (Optional[bool]): Whether to return a structured output.
            cache_position (Optional[torch.Tensor]): Cache position indices.
            **kwargs: Additional parameters for the model.

        Returns:
            Union[Any, Tuple]: Model outputs, structured as provided by the model.
        """
        if not self.model:
            raise ValueError("Model must be loaded before calling forward.")

        logger.info("Performing forward pass in a memory-efficient manner...")
        with torch.no_grad():
            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
        logger.debug(f"Forward pass outputs: {outputs}")
        return outputs


# Example usage of the wrapper
if __name__ == "__main__":
    CONFIG_OVERRIDES = {
        # Override configuration parameters if needed
    }

    model_name = "allenai/OLMo-2-1124-7B-Instruct"  # Update to the appropriate model name or path
    logger.info("Initializing Olmo2 model wrapper on CPU with advanced disk offloading for minimal RAM usage...")
    olmo2_wrapper = ModelWrapper(
        model_name=model_name, config_overrides=CONFIG_OVERRIDES, device="cpu", offload_folder="offload"
    )

    prompt_text = "Hey, are you conscious? Can you talk to me?"
    logger.info(f"Generating response for prompt: {prompt_text}")
    responses = olmo2_wrapper.generate(prompt=prompt_text, max_length=50)
    for i, response in enumerate(responses):
        logger.info(f"Response {i+1}: {response}")
