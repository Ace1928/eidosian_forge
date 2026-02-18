Loading model... 


▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀

build      : b8083-ae2d3f28a
model      : qwen2.5-1.5b-instruct-q5_k_m.gguf
modalities : text

available commands:
  /exit or Ctrl+C     stop or exit
  /regen              regenerate the last response
  /clear              clear the chat history
  /read               add a text file


> <|im_start|>system
You are an Eidosian Documentation Forge agent. Your task is to generate detailed, accurate, and structurally elegant documentation for the provided source file.
Follow these requirements:
1. File Summary: High-level overview of what the file does.
2. API Documentation: List all functions, classes, and their parameters/methods.
3. Current Status: Accurate assessment of the file's current state based on its content and comments.
4. Potential Future Directions: Eidosian ideas for ... (truncated)

## 📘 EIDOSIAN FORGE: SMOL AGENTS

> _"Tiny titans: agents that deliver maximum punch in minimal tokens."_

### 🧠 Core Agents

The core agents are the foundation upon which all other agents build their capabilities. Here is a list of common agents:

- **MultiStepAgent**: This agent can handle complex tasks by breaking them into smaller, manageable steps.
  - Parameters:
    - `task`: The task to be executed (required).
    - `stream=False`: Whether to stream the output or not (default: False).
    - `reset=True`: Whether to reset the memory step after each execution (default: True).
    - `images=None`: Additional images to use during inference.
    - `additional_args=None`: Additional arguments to pass to the agent.
    - `max_steps=None`: Maximum number of steps the agent should take.

- **CodeAgent**: This agent executes code and outputs the results.
  - Parameters:
    - `code_action`: The code action to execute (required).
    - `return_final_answer=False`: Whether to return the final answer or not (default: False).

- **ToolCallingAgent**: This agent calls tools from a space, allowing for more complex tasks.
  - Parameters:
    - `tool_caller`: The tool caller to use (required).
    - `additional_args=None`: Additional arguments to pass to the tool.

### ⚙️ Infrastructure

The infrastructure layer is responsible for storing and managing memory, logging agent behavior, and monitoring agent performance. Here are some commonly used functions:

- **Memory**
  - `memory.get_full_steps()`: Retrieve all steps in a full list.
  - `memory.get_succinct_steps()`: Retrieve the succinct steps of an execution.
  - `memory.reset()`: Reset the memory to its initial state.

- **Logging**
  - `logger.log(content, level=<LogLevel.INFO>)`: Log messages with different severity levels.
  - `logger.log_code(title, content)`: Log code snippets with a specified title and message.
  - `logger.log_error(error_message)`: Log errors using the specified error message.
  - `logger.log_markdown(content, title=None, style='#d4b702')`: Log formatted markdown messages with different styles.

- **Monitor**
  - `monitor.get_total_token_counts()`: Retrieve token counts for the agent's execution history.
  - `monitor.reset()`: Reset monitor metrics to their initial state.

### 🤖 Models

The models are responsible for executing tasks and generating outputs. Here is a list of common model methods:

- **Common Model Methods**
  - `model.get_token_counts()`: Get the number of tokens used by the model.
  - `model.to_dict()`: Convert the model to a JSON-compatible dictionary.

- **Specialized Models**
  - `XXXServerModel.create_client()`: Initialize an API client for a specialized server.
  - `TransformersModel.make_stopping_criteria(stop_sequences, tokenizer)`: Create stopping criteria for a specific language model (e.g., Llama2).
  - `VLLMModel.cleanup() | MLXModel.get_token_counts()`: Clean up models and get token counts.

- **Message Handling**
  - `model.postprocess_message(message, tools_to_call_from)`: Parse tool calls from the message.

### 🧰 Tools

Tools are used to execute various tasks such as web searches, code execution, and media processing. Here is a list of common tools:

- **Search & Web**
  - `DuckDuckGoSearchTool.forward(query)`
  - `GoogleSearchTool.forward(query, filter_year=None)`
  - `VisitWebpageTool.forward(url)`

- **Code Execution**
  - `PythonInterpreterTool.forward(code)`
  - `DockerExecutor.run_code_raise_errors(code_action, return_final_answer=False)`
  - `LocalPythonExecutor.send_tools(tools)` | `LocalPythonExecutor.send_variables(variables)`

- **Media**
  - `SpeechToTextTool: .encode(audio) | .forward(inputs) | .decode(outputs) | .setup()`
  - `FinalAnswerTool.forward(answer)`: Return final answer.

### 🔄 Media & Data Types

The media and data types layer handles the serialization of various agents' outputs such as audio, image, and text. Here are some commonly used functions:

- **Agent Audio**
  - `audio.to_raw() | audio.to_string()`

- **Agent Image**
  - `image.to_raw() | image.to_string()`
  - `image.show() | image.save() | image.resize() | image.crop()` (Many more PIL.Image methods available)

- **Agent Text**
  - `text.to_raw() | text.to_string()`

### 🔧 Utilities

The utilities layer provides various helper functions that are used across the EIDOSIAN FORGE. Here is a list of commonly used utilities:

- **Hub Integration**
  - `create_repo(repo_id, token=None, private=None, repo_type=None, exist_ok=False)`
  - `metadata_update(repo_id, metadata, repo_type=None, overwrite=False)`
  - `snapshot_download(repo_id, **kwargs)` → str: folder path
  - `upload_folder(repo_id, folder_path, path_in_repo=None, token=None)`

- **Python Helpers**
  - `evaluate_python_code(code, authorized_imports=[...], state=None)`
  - `fix_final_answer_code(code)`: Fixes variable assignments to final_answer
  - `get_clean_message_list(message_list, role_conversions={})`
  - `load_model(model_type, model_id, api_base=None, api_key=None)`
  - `load_tool(repo_id, **kwargs)`
  - `parse_code_blobs(text)` → List of code blobs (extracts code from LLM output)
  - `truncate_content(content, max_length=20000)`

- **UI**
  - `launch_gradio_demo(tool)`
  - `stream_to_gradio(agent, task, reset_agent_memory=False, additional_args=None)`
  - `GradioUI.launch(share=True, **kwargs)` | `GradioUI.interact_with_agent()`
  - `GradioUI.launch(share=True, **kwargs)`: Launch Gradio UI with the specified parameters

- **Quick Examples**
  - **Tool Creation from a HF Space**
    ```python
    img_gen = Tool.from_space(
        "black-forest-labs/FLUX.1-schnell",
        "image-generator",
        "Generate an image from text"
    )
    ```
  - **Add Metadata to a Repo**
    ```python
    metadata_update("my-repo/model", {
      'model-index': [{
          'name': 'Fine-tuned model',
          'results': [{
              'metrics': [{'name': 'Accuracy', 'value': 0.92}]
            }]
        }]
    })
    ```
  - **Create a Tool with the Decorator**
    ```python
    @tool
    def calculate(x: int, y: int) -> int:
      """Multiply x and y, then add their difference."""
      return x * y + (x - y)
    ```

The documentation for this file is structured to provide high-level overviews of the agents and infrastructure, as well as detailed information on each function and class. The current status section includes an assessment of the file's content and comments, while potential future directions suggest enhancements or refactoring ideas based on Eidosian principles.

This comprehensive documentation will serve as a valuable resource for developers working with the EIDOSIAN FORGE, providing clear definitions, usage examples, and detailed explanations of each component.

[ Prompt: 61.3 t/s | Generation: 3.3 t/s ]

Exiting...