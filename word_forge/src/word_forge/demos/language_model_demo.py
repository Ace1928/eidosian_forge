"""
Demonstration of the language model state and generation.
"""

from word_forge.parser.language_model import ModelState


def main():
    # Example usage
    model_state = ModelState()
    generated_text = model_state.generate_text(
        "Create an absolutely unhinged finish to this, keep it brief: Once upon a time",
        max_new_tokens=128,
    )
    print(generated_text)


if __name__ == "__main__":
    main()
