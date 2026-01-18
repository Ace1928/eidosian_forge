"""Simple text-based LLM training utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def train_model(
    files: Iterable[str],
    model_name: str = "distilgpt2",
    output_dir: str = "./trained_model",
    epochs: int = 1,
    batch_size: int = 1,
) -> None:
    """Train a causal language model on the provided text files.

    Args:
        files: Iterable of text file paths.
        model_name: Base model checkpoint from Hugging Face hub.
        output_dir: Directory to write the trained model.
        epochs: Number of training epochs.
        batch_size: Training batch size per device.
    """

    text_files: List[str] = [str(Path(f)) for f in files]
    logger.info("Loading dataset from %d files", len(text_files))
    dataset = load_dataset("text", data_files={"train": text_files})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=data_collator,
    )

    logger.info("Starting training for %d epochs", epochs)
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)
