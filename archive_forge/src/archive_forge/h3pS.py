from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "sshleifer/distilbart-cnn-12-6"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer to a directory
model.save_pretrained("/local_model_store/distilbart-cnn-12-6")
tokenizer.save_pretrained("/local_model_store/distilbart-cnn-12-6")


def summarize_text(text, model_path="./local_model_store/distilbart-cnn-12-6"):
    # Load the model and tokenizer from the local model store
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    # Create the summarization pipeline using the local model and tokenizer
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    # Perform summarization
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)

    # Extract and return the summary text
    return summary[0]["summary_text"]


# Example text
text = """
To use the `sshleifer/distilbart-cnn-12-6` model for text summarization and ensure that it loads from a local model store each time by default (minimizing or eliminating network requirements), you will need to follow these steps:

1. **Download the Model and Tokenizer**: First, download the model and tokenizer to your local machine. This step requires an initial network connection to download the files, but subsequent uses will not.

2. **Set Up the Code**: Modify your Python script to use the local copies of the model and tokenizer.

Here's how you can do it:

### Step 1: Download the Model and Tokenizer

You can download the model and tokenizer using the following commands in a Python script or directly in your Python environment:

### Step 2: Modify the Python Script

Now, modify your script to load the model and tokenizer from the local store by default. Here's how you can adjust your `textsummarygensim.py` script:

This script ensures that the model and tokenizer are loaded from the local directory specified by `model_path`, and the `local_files_only=True` parameter ensures that the Transformers library does not attempt to download files from the internet.

By following these steps, you can use the `sshleifer/distilbart-cnn-12-6` model for summarization while ensuring that it consistently loads from a local store, minimizing network requirements after the initial setup.
"""

# Get the summary
summary_text = summarize_text(text)
print("Summary:", summary_text)
