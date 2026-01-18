import os
import re
import json
import time
import logging
import datetime
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict, Union, cast

import requests
from requests.exceptions import RequestException, MissingSchema, HTTPError
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForQuestionAnswering # type: ignore
import wikipedia # type: ignore
from markdownify import markdownify # type: ignore
from duckduckgo_search import DDGS # type: ignore
from huggingface_hub import list_models # type: ignore
from huggingface_hub.utils import HfHubHTTPError # type: ignore
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

from smolagents import Tool, CodeAgent, tool, ToolCallingAgent, ManagedAgent, DuckDuckGoSearchTool

console = Console()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(threadName)s - %(message)s'
)

class HFModelDownloadsTool(Tool):
    """Retrieves the most downloaded models from the Hugging Face Hub."""
    name = "model_download_counter"
    description = """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint."""
    inputs = {
        "task": {
            "type": "string",
            "description": "the task category (such as text-classification, depth-estimation, etc)",
            "nullable": True,
        }
    }
    output_type = "string"

    _model_cache: Dict[Optional[str], Tuple[Any, float]] = {}
    CACHE_FILE = "hf_models_cache.json"
    CACHE_TTL = 3600

    def __init__(self):
        super().__init__()
        self._load_cache()

    def _load_cache(self):
        if not os.path.exists(self.CACHE_FILE):
            self._save_cache()
            return
        try:
            with open(self.CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                self._model_cache = {k: (v['data'], v['timestamp']) for k, v in cache_data.items()}
            console.print(f"Cache loaded from [bold blue]{self.CACHE_FILE}[/bold blue]")
        except Exception as e:
            console.print(f"[bold yellow]Error loading cache from {self.CACHE_FILE}: {e}[/bold yellow]")

    def _save_cache(self):
        try:
            with open(self.CACHE_FILE, 'w') as f:
                json.dump({k: {'data': v[0], 'timestamp': v[1]} for k, v in self._model_cache.items()}, f, indent=4)
            console.print(f"Cache saved to [bold blue]{self.CACHE_FILE}[/bold blue]")
        except Exception as e:
            console.print(f"[bold yellow]Error saving cache to {self.CACHE_FILE}: {e}[/bold yellow]")

    def forward(self, task: Optional[str] = None) -> str:
        if task in self._model_cache and time.time() - self._model_cache[task][1] < self.CACHE_TTL:
            console.print(f"Fetching from cache for task: '[bold green]{task}[/bold green]'")
            return self._model_cache[task][0]

        console.print(f"Fetching most downloaded model for task: '[bold green]{task}[/bold green]' from Hugging Face Hub")
        try:
            from huggingface_hub import list_models
            models = list(list_models(filter=task, sort="downloads", direction=-1))
            if models:
                most_downloaded_model_id = models[0].id
                console.print(f"Found most downloaded model for task '[bold green]{task}[/bold green]': [bold cyan]{most_downloaded_model_id}[/bold cyan]")
                self._model_cache[task] = (most_downloaded_model_id, time.time())
                self._save_cache()
                return most_downloaded_model_id
            else:
                error_message = f"No models found for task: {task}"
                self._model_cache[task] = (error_message, time.time())
                self._save_cache()
                return error_message
        except HfHubHTTPError as e:
            error_message = f"Hugging Face Hub API error fetching models for task '[bold red]{task}[/bold red]': {e}"
            console.print(f"[bold red]{error_message}[/bold red]")
            self._model_cache[task] = (error_message, time.time())
            self._save_cache()
            return error_message
        except Exception as e:
            error_message = f"Unexpected error fetching models for task '[bold red]{task}[/bold red]': {e}"
            console.print_exception()
            self._model_cache[task] = (error_message, time.time())
            self._save_cache()
            return error_message
        
model_downloads_tool = HFModelDownloadsTool()


class ImageGeneratorTool(Tool):
    """
    A tool for generating images from a text prompt using a Hugging Face Space.

    This tool leverages the `gradio-client` library (implicitly through `Tool.from_space`)
    to interact with a specified Hugging Face Space that offers image generation
    capabilities. It takes a textual description (prompt) as input, sends it to the
    remote Space, retrieves the generated image, and saves it to the local filesystem.

    Attributes:
        name (str): The name of the tool, used for identification and in agent prompts.
        description (str): A concise description of the tool's functionality, used in agent prompts.
        inputs (dict): Defines the expected input to the tool, specifying the 'prompt'
                       as a string with its own description.
        output_type (str): Specifies the output type of the tool, which is 'image' in this case.
        is_initialized (bool): A flag indicating whether the tool has been initialized.
    """
    name = "image_generator"
    description = "Generate an image from a prompt and save it to disk."
    inputs = {"prompt": {"type": "string", "description": "The prompt to generate an image from."}}
    output_type = "image"
    is_initialized: bool = False
    DEFAULT_SPACE_ID = "black-forest-labs/FLUX.1-schnell" # Define a constant for the default space ID

    def __init__(self, space_id: str = DEFAULT_SPACE_ID):
        """
        Initializes the ImageGeneratorTool.

        Establishes a connection to the specified Hugging Face Space using
        `Tool.from_space`.

        Args:
            space_id (str): The identifier of the Hugging Face Space to use for
                           image generation. This is typically in the format
                           "organization/space_name".
                           Defaults to "black-forest-labs/FLUX.1-schnell".

        Raises:
            ImportError: If the `gradio_client` library is not installed, as it's
                         required by `Tool.from_space`.
        """
        self.space_id = space_id
        # Create a Tool instance that wraps the Hugging Face Space. This leverages
        # the `gradio-client` library to interact with the Space's API.
        self._space_tool = Tool.from_space(
            self.space_id,
            name="image_generator_space_call",
            description="Helper tool to call the image generation space"
        )
        self.is_initialized = True

    def forward(self, prompt: str) -> Path:
        """
        Generates an image based on the provided text prompt.

        This method sends the prompt to the Hugging Face Space and retrieves the
        generated image. It then calls the `_save_image` method to save the
        image locally.

        Args:
            prompt (str): The textual description used to generate the image.

        Returns:
            Path: The file path to the saved image on the local filesystem.

        Raises:
            RuntimeError: If the underlying space tool fails to generate an image.
        """
        # Call the underlying space tool to generate the image. The `_space_tool`
        # instance handles the communication with the remote Hugging Face Space.
        try:
            image_path: str = self._space_tool(prompt) # type: ignore
        except NotImplementedError as e:
            raise RuntimeError(f"The Hugging Face Space tool returned an error: {e}") from e
        return self._save_image(image_path, prompt)

    def _save_image(self, image_path: str, prompt: str) -> Path:
        """
        Saves the generated image to the local filesystem.

        The image is saved in a directory named "images" within the current
        working directory. The filename is derived from the provided prompt,
        with spaces replaced by underscores and a maximum length to ensure
        filesystem compatibility. The image is saved in WebP format.

        Args:
            image_path (str): The path to the temporary image file returned by the
                              Hugging Face Space.
            prompt (str): The original text prompt used to generate the image.

        Returns:
            Path: The file path to the saved image on the local filesystem.

        Raises:
            FileNotFoundError: If the temporary image file from the Space cannot be found.
            OSError: If there is an error during file creation or copying.
        """
        # Create the images directory if it doesn't exist. `parents=True` ensures
        # that any necessary parent directories are also created. `exist_ok=True`
        # prevents an error if the directory already exists.
        Path("./images").mkdir(parents=True, exist_ok=True)

        # Generate a safe filename from the prompt. Convert to lowercase, replace
        # spaces with underscores, and limit the length to avoid overly long filenames.
        filename = prompt.lower().replace(" ", "_")[:50] + ".webp"

        # Construct the full path for the new image file within the "images" directory.
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') # Generate timestamp once
        new_image_path = Path("./images") / f"{timestamp}{filename}"
        # Copy the temporary image file to the new location. `shutil.copy2` preserves
        # metadata such as access and modification times.
        try:
            shutil.copy2(image_path, new_image_path)
            console.print(f"Image saved to [bold green]{new_image_path}[/bold green]") # Use rich formatting
            return new_image_path
        except FileNotFoundError as e:
            logging.error(f"Temporary image file not found: {e}", exc_info=True)
            raise
        except OSError as e:
            logging.error(f"Error saving image to {new_image_path}: {e}", exc_info=True)
            raise

image_generation_tool = ImageGeneratorTool()


@tool
def image_generator(prompt: str) -> str:
    """
    Generates an image based on the text prompt using a pre-trained text-to-image model or a Hugging Face Space.

    Args:
        prompt: The text prompt to generate the image from.

    Returns:
        The path to the generated image or a message indicating success or failure.
    """
    # Check if the prompt is a string
    if not isinstance(prompt, str):
        logging.error(f"TypeError: prompt must be a string, but got {type(prompt)}") # Add logging
        raise TypeError(f"prompt must be a string, but got {type(prompt)}")
    # Check if the prompt is empty
    if not prompt:
        logging.error("ValueError: prompt cannot be empty.") # Add logging
        raise ValueError("prompt cannot be empty.")

    try:
        # Use the advanced image generator tool
        image_path = image_generation_tool(prompt)
        return f"Image generated with prompt '{prompt}' at: {image_path}"
    except Exception as e:
        logging.error(f"Error during image generation: {e}", exc_info=True)
        return f"Error during image generation: {e}"

@tool
def wiki(query: str) -> str:
    """
    Retrieves information from Wikipedia.

    Args:
        query: The search query for Wikipedia.

    Returns:
        The content retrieved from Wikipedia.
    """
    # Check if the query is a string
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, but got {type(query)}")
    # Check if the query is empty
    if not query:
        raise ValueError("query cannot be empty.")
    try:
        # Search for the Wikipedia page
        page = wikipedia.page(query, auto_suggest=False)
        # Return the content of the page
        return page.content
    # Catch page not found errors
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'"
    # Catch disambiguation errors
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Wikipedia disambiguation error for '{query}'. Options: {e.options}"
    except Exception as e:
        logging.error(f"Error during wikipedia search: {e}", exc_info=True)
        return f"Error during wikipedia search: {e}"

@tool
def web_search(query: str) -> str:
    """
    Performs a web search using DuckDuckGo.

    Args:
        query: The search query.

    Returns:
        The search results as a string.
    """
    # Check if the query is a string
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, but got {type(query)}")
    # Check if the query is empty
    if not query:
        raise ValueError("query cannot be empty.")
    try:
        # Initialize DuckDuckGo search
        ddgs = DDGS()
        # Perform the search and get the results
        results = list(ddgs.text(query, max_results=5))
        # Format the results into a string
        return "\n".join([f"Title: {result['title']}\nBody: {result['body']}\nURL: {result['href']}" for result in results])
    # Catch any errors
    except Exception as e:
        logging.error(f"Error during web search: {e}", exc_info=True)
        return f"Error during web search: {e}"

@tool
def visit_webpage(url: str) -> str:
    """
    Retrieves the content of a webpage at the given URL and returns it as a Markdown-formatted string.

    This tool sends an HTTP GET request to the specified URL. Upon successful retrieval, the HTML
    content is converted to Markdown format, with multiple consecutive line breaks condensed to a maximum of two.

    Args:
        url: The URL of the webpage to visit. Must be a valid URL starting with 'http://' or 'https://'.

    Returns:
        str: The Markdown-formatted content of the webpage. If the retrieval fails, returns an error message as a string.

    Raises:
        TypeError: If the `url` argument is not a string.
        ValueError: If the `url` argument is an empty string.
        requests.exceptions.MissingSchema: If the URL is improperly formatted (e.g., missing 'http://' or 'https://').
        requests.exceptions.RequestException: If there's an issue with the HTTP request, such as network errors, timeouts, or invalid URLs.
        Exception: For any other unexpected errors during the process of fetching or processing the webpage content.
    """
    # Check if the URL is a string
    if not isinstance(url, str):
        logging.error(f"TypeError: URL must be a string, but got {type(url)}")
        raise TypeError(f"URL must be a string, but got {type(url)}")
    # Check if the URL is empty
    if not url:
        logging.error("ValueError: URL cannot be empty.")
        raise ValueError("URL cannot be empty.")

    try:
        # Log the URL being fetched
        logging.debug(f"Fetching content from URL: '{url}'")
        # Send a GET request to the URL with a timeout of 15 seconds
        response = requests.get(url, timeout=15)
        # Raise an HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status()

        # Convert the HTML content to Markdown format and remove leading/trailing whitespace
        markdown_content = markdownify(response.text).strip()
        # Condense multiple line breaks for cleaner output
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        # Log successful fetch and conversion
        logging.info(f"Successfully fetched and converted content from: {url}")
        # Return the markdown content
        return markdown_content

    except MissingSchema as e:
        logging.error(f"Invalid URL format: '{url}'. Error: {e}")
        return f"Error: Invalid URL format. Please provide a valid URL starting with 'http://' or 'https://'."
    except HTTPError as e:
        log_message = f"HTTP error fetching URL '{url}': {e}" # Define log message once
        logging.error(log_message)
        return f"Error fetching URL '{url}': HTTP error occurred with status code {e.response.status_code}."
    except RequestException as e:
        logging.error(f"Error fetching URL '{url}': {e}", exc_info=True)
        return f"Error fetching the webpage at '{url}'. Please check the URL and your network connection."
    except Exception as e:
        logging.exception(f"Unexpected error while processing URL '{url}': {e}")
        return f"An unexpected error occurred while processing the URL '{url}': {e}"
    
@tool
def final_answer(answer: str) -> str:
    """
    Returns the final answer after analysis and improvement.

    Args:
        answer: The final answer to be returned.

    Returns:
        The final improved answer as a string.
    """
    if not isinstance(answer, str):
        logging.error(f"TypeError: Answer must be a string, but got {type(answer)}")
        raise TypeError(f"Answer must be a string, but got {type(answer)}")
    if not answer:
        logging.error("ValueError: Answer cannot be empty.")
        raise ValueError("Answer cannot be empty.")

    try:
        model_path = Path("saved_models/Qwen/Qwen2.5-0.5B-Instruct")
        if not model_path.exists():
            error_message = f"Translation model not found at '{model_path}'"
            logging.error(error_message)
            raise FileNotFoundError(error_message)

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(str(model_path))

        # --- Analysis and Improvement Phase ---
        analysis_prompt = (
            f"This is your current answer: {answer}\n\n"
            "Please ensure you reiterate over it and analyse it and provide 3 key impactful, "
            "significant, but implementable improvements to the response. Output only the 3 improvements."
        )
        inputs = tokenizer(analysis_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512)
        improvement_plan = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Improvement plan: {improvement_plan}")

        # --- Apply Improvements Phase ---
        improvement_prompt = (
            f"Original Answer: {answer}\n\n"
            f"Improvement Plan: {improvement_plan}\n\n"
            "Based on the improvement plan, please provide the complete improved final answer in its entirety."
        )
        inputs = tokenizer(improvement_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=1024)
        improved_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Improved answer: {improved_answer}")

        return improved_answer

    except Exception as e:
        logging.exception(f"Error during final answer processing: {e}")
        return f"An error occurred during final answer processing: {e}"

@tool
def translator(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translates text from one language to another using a pre-trained translation model.

    Args:
        text: The text to be translated.
        src_lang: The source language of the text.
        tgt_lang: The target language for translation.

    Returns:
        The translated text.
    """
    # Check if the text is a string
    if not isinstance(text, str):
        logging.error(f"TypeError: text must be a string, but got {type(text)}")
        raise TypeError(f"text must be a string, but got {type(text)}")
    # Check if the source language is a string
    if not isinstance(src_lang, str):
        logging.error(f"TypeError: src_lang must be a string, but got {type(src_lang)}")
        raise TypeError(f"src_lang must be a string, but got {type(src_lang)}")
    # Check if the target language is a string
    if not isinstance(tgt_lang, str):
        logging.error(f"TypeError: tgt_lang must be a string, but got {type(tgt_lang)}")
        raise TypeError(f"tgt_lang must be a string, but got {type(tgt_lang)}")
    # Check if the text is empty
    if not text:
        logging.error("ValueError: text cannot be empty.")
        raise ValueError("text cannot be empty.")
    # Check if the source language is empty
    if not src_lang:
        logging.error("ValueError: src_lang cannot be empty.")
        raise ValueError("src_lang cannot be empty.")
    # Check if the target language is empty
    if not tgt_lang:
        logging.error("ValueError: tgt_lang cannot be empty.")
        raise ValueError("tgt_lang cannot be empty.")

    try:
        # Define the relative path to the saved model
        model_path = Path("saved_models/Qwen/Qwen2.5-0.5B-Instruct")
        if not model_path.exists():
            error_message = f"Translation model not found at '{model_path}'" # Define error message once
            logging.error(error_message)
            raise FileNotFoundError(error_message)

        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(str(model_path))

        # Save the model if it hasn't been saved yet
        if not (model_path / "pytorch_model.bin").exists():
            model.save_pretrained(model_path, safe_serialization=True)
            tokenizer.save_pretrained(model_path, safe_serialization=True)
            console.print(f"Model and tokenizer saved to [bold blue]{model_path}[/bold blue]")


        # Prepare the prompt for translation
        prompt = f"can you please translate '{text}' into {tgt_lang} and output the {tgt_lang} translation directly, verbatim, in its entirety."

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate the translation
        outputs = model.generate(**inputs, max_new_tokens=200)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Improve the translated text using the final_answer tool
        if isinstance(translated_text, str):
            try:
                improved_translation = final_answer(translated_text)
            except NotImplementedError:
                improved_translation = "Translation improvement not implemented."
            return str(improved_translation)
        else:
            raise ValueError("Translated text is not a string.")

    except Exception as e:
        logging.error(f"Error during translation: {e}", exc_info=True)
        return f"Error during translation: {e}"
    
@tool
def document_qa(document: str, question: str) -> str:
    """
    Answers a question based on the provided document using a pre-trained QA model.

    Args:
        document: The document content to answer the question from.
        question: The question to answer.

    Returns:
        The answer to the question.
    """
    # Check if the document is a string
    if not isinstance(document, str):
        logging.error(f"TypeError: document must be a string, but got {type(document)}")
        raise TypeError(f"document must be a string, but got {type(document)}")
    # Check if the question is a string
    if not isinstance(question, str):
        logging.error(f"TypeError: question must be a string, but got {type(question)}")
        raise TypeError(f"question must be a string, but got {type(question)}")
    # Check if the document is empty
    if not document:
        logging.error("ValueError: document cannot be empty.")
        raise ValueError("document cannot be empty.")
    # Check if the question is empty
    if not question:
        logging.error("ValueError: question cannot be empty.")
        raise ValueError("question cannot be empty.")

    try:
        # Define the relative path to the saved model
        model_path = Path("saved_models/Qwen/Qwen2.5-0.5B-Instruct")

        if not model_path.exists():
            error_message = f"QA model not found at '{model_path}'"
            logging.error(error_message)
            raise FileNotFoundError(error_message)

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForQuestionAnswering.from_pretrained(str(model_path))

        inputs = tokenizer(question, document, add_special_tokens=True, return_tensors="pt")
        input_ids: List[int] = inputs.input_ids.tolist()[0]

        output = model(**inputs)
        answer_start_scores = output.start_logits
        answer_end_scores = output.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        # Ensure answer_start and answer_end are integers
        token_ids = input_ids[int(answer_start):int(answer_end)]
        token_list = cast(List[str], tokenizer.convert_ids_to_tokens(token_ids))
        answer: str = tokenizer.convert_tokens_to_string(token_list)

        return answer
    except Exception as e:
        logging.error(f"Error during document QA: {e}", exc_info=True)
        return f"Error during document QA: {e}"

if __name__ == '__main__':
    # Test HFModelDownloadsTool
    try:
        most_downloaded_text_model = HFModelDownloadsTool().forward("text-generation")
        print(f"Most downloaded text generation model: {most_downloaded_text_model}")
    except Exception as e:
        print(f"Error testing HFModelDownloadsTool: {e}")

    # Test visit_webpage
    try:
        webpage_content = visit_webpage("https://www.wikipedia.org/wiki/Python_(programming_language)")
        if isinstance(webpage_content, str):
            print(f"Webpage content (truncated):\n{webpage_content[:1000]}...")
        else:
            print(f"Unexpected result type: {type(webpage_content)}")
    except Exception as e:
        print(f"Error testing visit_webpage: {e}")

    # Test document_qa
    try:
        document = "The quick brown fox jumps over the lazy dog."
        question = "What does the fox jump over?"
        try:
            answer = document_qa(str(document), str(question))
            print(f"Answer to question: {answer}")
        except NotImplementedError as e:
            print(f"Error testing document_qa: {e}")
        except Exception as e:
            print(f"Error testing document_qa: {e}")
    except Exception as e:
        print(f"Error testing document_qa: {e}")

    # Test image_generator
    try:
        prompt = "A sunny beach"
        image_path = image_generator(prompt)
        print(f"Image generated with image_generator at: {image_path}")
    except Exception as e:
        print(f"Error testing image_generator: {e}")

    # Test image_generation_tool directly
    try:
        prompt = "A cat wearing a hat"
        image_path = image_generation_tool(prompt)
        print(f"Image generated with image_generation_tool at: {image_path}")
    except Exception as e:
        print(f"Error testing image_generation_tool: {e}")

    # Test final_answer
    final = final_answer("This is the final answer.")
    print(f"Final answer: {final}")

    # Test translator
    try:
        translated_text = translator("Hello, how are you?", "en", "es")
        print(f"Translated text: {translated_text}")
    except Exception as e:
        print(f"Error testing translator: {e}")

    # Test wiki
    try:
        wiki_content = wiki("Python (programming language)")
        if isinstance(wiki_content, str):
            print(f"Wiki content (truncated):\n{wiki_content[:1000]}...")
        else:
            print(f"Unexpected result type: {type(wiki_content)}")
    except Exception as e:
        print(f"Error testing wiki: {e}")

    # Test web_search
    try:
        search_results = web_search("artificial intelligence")
        if isinstance(search_results, str):
            print(f"Web search results (truncated):\n{search_results[:1000]}...")
        else:
            print(f"Unexpected result type: {type(search_results)}")
    except Exception as e:
        print(f"Error testing web_search: {e}")
