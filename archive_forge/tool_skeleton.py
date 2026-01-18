from smolagents.tools import Tool, tool
from typing import Dict, Any, Callable, Type
from pydantic import BaseModel

# ##############################################################################
# Defining Tools in smolagents: A Comprehensive Guide
# ##############################################################################
# This file provides a detailed guide to defining custom tools for use with
# smolagents. Tools are essential for extending the capabilities of agents,
# enabling them to interact with the external world and perform specific tasks.

# There are two primary methods for defining tools in smolagents, each suited
# to different levels of complexity and requirements:

# 1. Using the `@tool` decorator for simple, stateless functions. This is the
#    recommended approach for straightforward tools that perform a single action.

# 2. Subclassing the `Tool` class for more complex tools that may require
#    internal state, multiple methods, or finer control over their behavior.

# ##############################################################################
# Method 1: Defining Tools using the @tool Decorator
# ##############################################################################
# The `@tool` decorator provides a concise way to define simple, stateless tools.
# It automatically handles the creation of necessary metadata.

# Key Features:
# - Simplicity: Ideal for tools that perform a single, well-defined action.
# - Stateless: Typically used for functions that do not need to maintain internal state.
# - Automatic Metadata: The decorator automatically extracts name and description from the function signature and docstring.

# Example:
# ```python
# import requests
# from smolagents.tools import tool
# import re
# from markdownify import markdownify
# import logging

# @tool
# def visit_webpage(url: str) -> str:
#     """
#     Retrieves the content of a webpage at the given URL and returns it as Markdown.

#     Args:
#         url: The URL of the webpage to visit.

#     Returns:
#         The Markdown-formatted content of the webpage.
#     """
#     if not isinstance(url, str):
#         logging.error(f"TypeError: URL must be a string, but got {type(url)}")
#         raise TypeError(f"URL must be a string, but got {type(url)}")
#     if not url:
#         logging.error("ValueError: URL cannot be empty.")
#         raise ValueError("URL cannot be empty.")

#     try:
#         logging.debug(f"Fetching content from URL: '{url}'")
#         response = requests.get(url, timeout=15)
#         response.raise_for_status()

#         markdown_content = markdownify(response.text).strip()
#         markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

#         logging.info(f"Successfully fetched and converted content from: {url}")
#         return markdown_content

#     except requests.exceptions.MissingSchema as e:
#         logging.error(f"Invalid URL format: '{url}'. Error: {e}")
#         return f"Error: Invalid URL format. Please provide a valid URL starting with 'http://' or 'https://'."
#     except requests.exceptions.HTTPError as e:
#         logging.error(f"HTTP error fetching URL '{url}': {e}")
#         return f"Error fetching URL '{url}': HTTP error occurred with status code {e.response.status_code}."
#     except requests.exceptions.RequestException as e:
#         logging.error(f"Error fetching URL '{url}': {e}", exc_info=True)
#         return f"Error fetching the webpage at '{url}'. Please check the URL and your network connection."
#     except Exception as e:
#         logging.exception(f"Unexpected error while processing URL '{url}': {e}")
#         return f"An unexpected error occurred while processing the URL '{url}': {e}"
# ```

# **Key Components for Decorated Tools:**
#
# - **`@tool` decorator:**  Marks the function as a smolagents tool, enabling its discovery and use by agents.
# - **`your_tool_name`:** The name of your tool. This should be unique and descriptive, clearly indicating the tool's purpose. The function name itself serves as the tool's name.
# - **Docstring:** A comprehensive docstring is crucial. It should explain what the tool does, its arguments, and what it returns. This is used by the agent to understand the tool's functionality.
# - **Type Hints:** Use clear and accurate type hints for all input arguments and the return value. This improves code readability and allows for better validation.
# - **Imports:**  For tools intended to be shared on the Hub, include all necessary import statements *within* the function. This ensures the tool is self-contained and can be executed in different environments.

@tool
def my_decorated_tool(text_input: str) -> str:
    """
    A simple example tool that converts input text to uppercase.

    Args:
        text_input: The text to be converted.

    Returns:
        The input text in uppercase.
    """
    processed_text = text_input.upper()
    return processed_text

# ##############################################################################
# Method 2: Defining Tools by Subclassing the Tool Class
# ##############################################################################
# Subclassing the `Tool` class offers greater flexibility for creating complex
# tools. This approach is suitable for tools that require internal state,
# multiple methods, or more intricate logic.

# Key Features:
# - State Management: Allows the tool to maintain internal state across multiple calls.
# - Multiple Methods: Enables the organization of tool logic into several methods for better clarity and modularity.
# - Custom Initialization: Provides control over the tool's initialization process.

# Example 1: Image Generation Tool
# ```python
# import os
# from pathlib import Path
# from smolagents.tools import Tool
# import shutil
# import datetime

# class ImageGeneratorTool(Tool):
#     """Generates images from a text prompt using a Hugging Face Space."""
#     name = "image_generator"
#     description = "Generate an image from a prompt and save it to disk."
#     inputs = {"prompt": {"type": "string", "description": "The prompt to generate an image from."}}
#     output_type = "image"
#     is_initialized: bool = False

#     def __init__(self, space_id: str = "black-forest-labs/FLUX.1-schnell"):
#         super().__init__()
#         from smolagents.tools import Tool  # Import here for Hub compatibility
#         self.space_id = space_id
#         self._space_tool = Tool.from_space(
#             self.space_id,
#             name="image_generator_space_call",
#             description="Helper tool to call the image generation space"
#         )
#         self.is_initialized = True

#     def forward(self, prompt: str) -> Path:
#         image_path = self._space_tool(prompt)
#         return self._save_image(image_path, prompt)

#     def _save_image(self, image_path: str, prompt: str) -> Path:
#         Path("./images").mkdir(parents=True, exist_ok=True)
#         filename = prompt.lower().replace(" ", "_")[:50] + ".webp"
#         new_image_path = Path("./images") / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_')}{filename}"
#         shutil.copy2(image_path, new_image_path)
#         print(f"Image saved to {new_image_path}")
#         return str(new_image_path)
# ```

# Example 2: Hugging Face Model Downloads Tool
# ```python
# import json
# import os
# import time
# from typing import Dict, List, Any, Optional, Tuple

# from huggingface_hub import list_models
# from huggingface_hub.utils import HfHubHTTPError
# from rich.console import Console
# from smolagents import Tool

# console = Console()

# class HFModelDownloadsTool(Tool):
#     """Retrieves the most downloaded models from the Hugging Face Hub."""
#     name = "model_download_counter"
#     description = "Retrieves the most downloaded model for a given task from the Hugging Face Hub."
#     inputs = {
#         "task": {
#             "type": "string",
#             "description": "The task category (e.g., text-classification).",
#             "nullable": True,
#         }
#     }
#     output_type = "string"

#     _model_cache: Dict[Optional[str], Tuple[Any, float]] = {}
#     CACHE_FILE = "hf_models_cache.json"
#     CACHE_TTL = 3600

#     def __init__(self):
#         super().__init__()
#         self._load_cache()

#     def _load_cache(self):
#         if os.path.exists(self.CACHE_FILE):
#             try:
#                 with open(self.CACHE_FILE, 'r') as f:
#                     cache_data = json.load(f)
#                     self._model_cache = {k: (v['data'], v['timestamp']) for k, v in cache_data.items()}
#                 console.print(f"Cache loaded from [bold blue]{self.CACHE_FILE}[/bold blue]")
#             except Exception as e:
#                 console.print(f"[bold yellow]Error loading cache from {self.CACHE_FILE}: {e}[/bold yellow]")

#     def _save_cache(self):
#         try:
#             with open(self.CACHE_FILE, 'w') as f:
#                 json.dump({k: {'data': v[0], 'timestamp': v[1]} for k, v in self._model_cache.items()}, f, indent=4)
#             console.print(f"Cache saved to [bold blue]{self.CACHE_FILE}[/bold blue]")
#         except Exception as e:
#             console.print(f"[bold yellow]Error saving cache to {self.CACHE_FILE}: {e}[/bold yellow]")

#     def forward(self, task: Optional[str] = None) -> str:
#         if task in self._model_cache and time.time() - self._model_cache[task][1] < self.CACHE_TTL:
#             console.print(f"Fetching from cache for task: '[bold green]{task}[/bold green]'")
#             return self._model_cache[task][0]

#         console.print(f"Fetching most downloaded model for task: '[bold green]{task}[/bold green]' from Hugging Face Hub")
#         try:
#             models: List[Any] = list(list_models(filter=task, sort="downloads", direction=-1))
#             if models:
#                 most_downloaded_model_id = models[0].id
#                 console.print(f"Found most downloaded model for task '[bold green]{task}[/bold green]': [bold cyan]{most_downloaded_model_id}[/bold cyan]")
#                 self._model_cache[task] = (most_downloaded_model_id, time.time())
#                 self._save_cache()
#                 return most_downloaded_model_id
#             else:
#                 error_message = f"No models found for task: {task}"
#                 self._model_cache[task] = (error_message, time.time())
#                 self._save_cache()
#                 return error_message
#         except HfHubHTTPError as e:
#             error_message = f"Hugging Face Hub API error fetching models for task '[bold red]{task}[/bold red]': {e}"
#             console.print(f"[bold red]{error_message}[/bold red]")
#             self._model_cache[task] = (error_message, time.time())
#             self._save_cache()
#             return error_message
#         except Exception as e:
#             error_message = f"Unexpected error fetching models for task '[bold red]{task}[/bold red]': {e}"
#             console.print_exception()
#             self._model_cache[task] = (error_message, time.time())
#             self._save_cache()
#             return error_message
# ```

# **Key Components for Class-Based Tools:**
#
# - **Class Definition:** Your tool class must inherit from `smolagents.tools.Tool`.
# - **`name` (str):**  A unique and descriptive name for the tool. This name is used by the agent to identify and call the tool. Choose a name that clearly reflects the tool's function (e.g., "image_generator", "huggingface_model_downloads").
# - **`description` (str):** A concise explanation of what the tool does. This description is included in the agent's system prompt, helping the agent understand when to use the tool.
# - **`inputs` (Dict[str, Dict[str, str]]):** Defines the input parameters the tool expects. This is a dictionary where each key is the name of an input, and the value is another dictionary specifying the input's "type" (a Pydantic-compatible type like "string", "integer", "boolean") and a "description" of the input.
# - **`output_type` (str):** Specifies the type of the output the tool will return. This should also be a Pydantic-compatible type.
# - **`__init__(self)`:** The constructor for your tool. For tools intended to be shared on the Hub, the `__init__` method should ideally only accept `self` as an argument. Any necessary configuration should be handled via class attributes.
# - **`forward(self, ...)`:** This method contains the core logic of your tool. It accepts the inputs defined in the `inputs` dictionary as arguments (with appropriate type hints) and returns a value of the type specified in `output_type`.
# - **Imports:**  Include all necessary import statements *within* the `forward` method (or other methods where they are used) for tools intended for Hub sharing. This ensures the tool is self-contained.

class AdvancedTool(Tool):
    """
    A more advanced tool demonstrating subclassing of the `Tool` class.
    """
    name: str = "advanced_tool"
    description: str = "Performs an advanced operation on the input."
    inputs = {
        "data": {"type": "string", "description": "The primary data to process."},
        "multiplier": {"type": "integer", "description": "A number to multiply with the data length."}
    }
    output_type: str = "string"

    def __init__(self):
        super().__init__()
        self.internal_counter = 0

    def forward(self, data: str, multiplier: int) -> str:
        """Executes the core logic of the AdvancedTool."""
        self.internal_counter += 1
        result = f"Advanced tool processed '{data}' with multiplier {multiplier}. Counter: {self.internal_counter}"
        return result

# ##############################################################################
# Key Considerations for Defining Effective and Shareable Tools
# ##############################################################################
# When creating your tools, keep the following best practices in mind:
#
# - **Unique and Descriptive Name:** Choose a name that clearly and concisely indicates the tool's function. This helps agents (and users) understand what the tool does.
# - **Clear Description:** Provide a brief but informative description of the tool's purpose. This is crucial for the agent to determine when to use the tool.
# - **Comprehensive Docstrings:** Document the purpose, arguments, and return values for both decorated functions and the `forward` method of class-based tools. Good documentation is essential for usability and maintainability.
# - **Well-Defined Inputs:** For class-based tools, use the `inputs` dictionary to explicitly define the name, Pydantic type, and description of each input parameter. This ensures the agent provides the correct information.
# - **Explicit Output Type:** Define the `output_type` using a Pydantic-compatible type. This allows for validation of the tool's output.
# - **Type Hints:** Utilize type hints for all arguments and return values in both decorated functions and class methods. This enhances code clarity and enables static analysis.
# - **Pydantic Compatibility:** Ensure that the types specified in `inputs` and `output_type` are valid Pydantic data types. This ensures seamless integration with the smolagents framework.
# - **Self-Contained Imports:** For tools intended to be shared on the Hub, import all necessary libraries *within* the decorated function or the `forward` method of a class-based tool. This makes the tool portable and easy to use in different environments.
# - **`__init__` Method Best Practices:** When subclassing `Tool` for Hub sharing, the `__init__` method should ideally only accept `self`. Use class attributes for any fixed configurations. This ensures compatibility when sharing on the Hub.
# - **Focused `forward` Method:** The `forward` method should contain the core logic of the tool, taking the defined inputs and returning the specified `output_type`. Keep this method focused and well-organized.

# ##############################################################################
# Sharing Your Tool on the Hub
# ##############################################################################
# Sharing your custom tools on the Hugging Face Hub makes them accessible to the
# wider community. Use the `push_to_hub()` method to share your tools:
#
# ```python
# # For a class-based tool:
# image_generation_tool_instance = ImageGeneratorTool()
# image_generation_tool_instance.push_to_hub("your_username/your-image-tool", token="YOUR_HUGGINGFACE_API_TOKEN")
#
# # For a decorated tool:
# my_decorated_tool.push_to_hub("your_username/my-simple-tool", token="YOUR_HUGGINGFACE_API_TOKEN")
# ```
#
# **Before pushing, ensure:**
# - You have created a repository on the Hugging Face Hub where you want to host your tool.
# - You have a Hugging Face API token with write access to the repository.
# - Your tool adheres to the Hub compatibility rules:
#     - All necessary imports are within the tool's methods (for class-based tools, primarily within `forward`).
#     - If you subclass the `__init__` method, ensure it only accepts `self` as an argument. Use class attributes for any fixed configurations.

# ##############################################################################
# Loading Tools from the Hub
# ##############################################################################
# You can easily load tools shared on the Hub using the `Tool.from_hub()` or
# `load_tool()` methods:
#
# ```python
# from smolagents.tools import load_tool
#
# loaded_tool = load_tool("your_username/your-tool-repository", trust_remote_code=True)
# ```
#
# **Important:** The `trust_remote_code=True` argument is crucial when loading tools from the Hub. Since loading tools involves executing code potentially written by others, you should only load tools from repositories you trust.
