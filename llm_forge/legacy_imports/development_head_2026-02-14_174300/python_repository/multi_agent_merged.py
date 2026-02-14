import os
import logging
import re
import huggingface_hub
from networkx import complement
import requests
import json
import time
import hashlib
import psutil
import subprocess
from typing import Dict, Optional, List, Any
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from urllib.parse import urlparse
from markdownify import markdownify
from requests.exceptions import RequestException
from dotenv import load_dotenv
import huggingface_hub

# smolagents imports
from smolagents import tool, CodeAgent, ToolCallingAgent, HfApiModel, ManagedAgent, DuckDuckGoSearchTool, Tool

###############################################################################
#                               LOGGING SETUP                                 #
###############################################################################
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_handler: logging.StreamHandler = logging.StreamHandler()
log_formatter: logging.Formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

###############################################################################
#                         LOAD ENVIRONMENT VARIABLES                          #
###############################################################################
load_dotenv()
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
if HF_TOKEN:
    logger.info("HF_TOKEN loaded from .env file.")
else:
    logger.warning("HF_TOKEN not found in .env file.")

huggingface_hub.login(token='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')

###############################################################################
#                              DISK IO MANAGEMENT                             #
###############################################################################
class DiskIOManager:
    """
    Manages disk-based read and write operations in chunks to handle large files efficiently.
    """

    def __init__(self, chunk_size: int = 1024 * 1024) -> None:
        """
        Initializes the DiskIOManager with a specified chunk size.

        Args:
            chunk_size (int): The size in bytes for reading/writing chunks from/to disk.
                              Defaults to 1MB.
        """
        self.chunk_size: int = chunk_size
        logger.debug(f"DiskIOManager initialized with chunk size: {self.chunk_size} bytes.")

    def write_data_to_disk(self, file_path: str, data: str, mode: str = "w") -> None:
        """
        Writes data (string) to disk in chunks.

        Args:
            file_path (str): The path to the file where data will be written.
            data (str): The string data to write to the file.
            mode (str): The file opening mode (e.g., 'w' for write, 'a' for append).
                        Defaults to 'w'.

        Returns:
            None
        """
        logger.debug(f"Writing to disk: {file_path}, mode: {mode}")
        try:
            with open(file_path, mode, encoding="utf-8") as f:
                for i in range(0, len(data), self.chunk_size):
                    f.write(data[i : i + self.chunk_size])
            logger.debug(f"Successfully wrote data to {file_path}")
        except IOError as e:
            logger.error(f"Error writing to {file_path}: {e}", exc_info=True)
            raise

    def read_data_from_disk(self, file_path: str, mode: str = "r") -> str:
        """
        Reads data from disk in chunks and returns it as a string.

        Args:
            file_path (str): The path to the file to read.
            mode (str): The file opening mode (e.g., 'r' for read). Defaults to 'r'.

        Returns:
            str: The content read from the file.
        """
        logger.debug(f"Reading from disk: {file_path}, mode: {mode}")
        try:
            data: str = ""
            with open(file_path, mode, encoding="utf-8") as f:
                while True:
                    chunk: str = f.read(self.chunk_size)
                    if not chunk:
                        break
                    data += chunk
            logger.debug(f"Successfully read data from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}", exc_info=True)
            raise
        except IOError as e:
            logger.error(f"Error reading from {file_path}: {e}", exc_info=True)
            raise

    def read_data_from_disk_in_chunks(self, file_path: str, mode: str = "r") -> Any:
        """
        Generator that yields data from disk in chunks for memory efficiency.

        Args:
            file_path (str): The path to the file to read in chunks.
            mode (str): The file opening mode. Defaults to 'r'.

        Yields:
            str: A chunk of data read from the file.
        """
        logger.debug(f"Reading data in chunks: {file_path}, mode: {mode}")
        try:
            with open(file_path, mode, encoding="utf-8") as f:
                while True:
                    chunk: str = f.read(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except FileNotFoundError:
            logger.error(f"File not found in chunk reader: {file_path}", exc_info=True)
            raise
        except IOError as e:
            logger.error(f"Error reading from {file_path}: {e}", exc_info=True)
            raise

###############################################################################
#                           ADAPTIVE THREAD POOL                               #
###############################################################################
class AdaptiveThreadPool:
    """
    Manages a thread pool that dynamically adjusts its size based on CPU usage.
    """

    def __init__(self, base_workers: int = 2, max_workers: int = 10) -> None:
        """
        Initializes the AdaptiveThreadPool with base and maximum worker counts.

        Args:
            base_workers (int): The initial number of worker threads. Defaults to 2.
            max_workers (int): The maximum number of worker threads the pool can scale up to. Defaults to 10.
        """
        self.base_workers: int = base_workers
        self.max_workers: int = max_workers
        self._pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=self.base_workers)
        logger.info(f"AdaptiveThreadPool initialized with {self.base_workers} base workers.")

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Submits a task to the thread pool.

        Args:
            fn: The function to execute.
            *args: Positional arguments for fn.
            **kwargs: Keyword arguments for fn.

        Returns:
            Future: A concurrent.futures.Future representing the execution of the task.
        """
        return self._pool.submit(fn, *args, **kwargs)

    def adjust_pool_size(self) -> None:
        """
        Dynamically adjusts the thread pool size based on CPU usage.
        """
        cpu_percent: float = psutil.cpu_percent() / 100
        current_workers: int = self._pool._max_workers
        target_workers: int = min(
            self.max_workers,
            max(self.base_workers, round(self.max_workers * (1 - cpu_percent)))
        )
        if target_workers != current_workers:
            logger.info(
                f"Adjusting thread pool size from {current_workers} to {target_workers} "
                f"based on CPU usage ({cpu_percent:.2f})."
            )
            self._pool.shutdown(wait=False)
            self._pool = ThreadPoolExecutor(max_workers=target_workers)

    def map(self, func: Any, iterable: Any, timeout: Optional[float] = None, chunksize: int = 1) -> Any:
        """
        Submits multiple tasks to the thread pool and returns an iterator of results.

        Args:
            func: The function to apply to each item in the iterable.
            iterable: An iterable of items to process.
            timeout (Optional[float]): Maximum number of seconds to wait for results.
            chunksize (int): The size of the chunks the iterable will be split into.

        Returns:
            iterator: An iterator that yields the function results.
        """
        return self._pool.map(func, iterable, timeout=timeout, chunksize=chunksize)

    def shutdown(self, wait: bool = True) -> None:
        """
        Shuts down the thread pool.

        Args:
            wait (bool): If True, waits for all tasks to complete before returning.
        """
        self._pool.shutdown(wait=wait)
        logger.info("AdaptiveThreadPool shut down.")

###############################################################################
#                          METRICS AND LOGGING TOOLS                          #
###############################################################################
METRICS_LOG_FILE: str = "tool_metrics.log"

def log_tool_metrics(
    tool_name: str,
    start_time: float,
    end_time: float,
    inputs: Dict[str, Any],
    outputs: str,
    error: Optional[str] = None,
) -> None:
    """
    Logs detailed metrics for each tool execution to a centralized metrics file.

    Args:
        tool_name (str): The name of the tool function.
        start_time (float): The timestamp when the tool execution started.
        end_time (float): The timestamp when the tool execution ended.
        inputs (Dict[str, Any]): The dictionary of inputs passed to the tool function.
        outputs (str): The output produced by the tool function.
        error (Optional[str]): The error message if any error occurred; otherwise None.

    Returns:
        None
    """
    duration: float = end_time - start_time
    metrics: Dict[str, Any] = {
        "tool_name": tool_name,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "inputs": inputs,
        "outputs": outputs,
        "error": error,
    }
    try:
        os.makedirs(os.path.dirname(METRICS_LOG_FILE), exist_ok=True)
    except (FileNotFoundError, OSError):
        pass

    try:
        with open(METRICS_LOG_FILE, "a", encoding="utf-8") as f:
            json.dump(metrics, f)
            f.write("\n")
        logger.debug(f"Metrics for '{tool_name}' logged to {METRICS_LOG_FILE} (duration={duration:.2f}s)")
    except IOError as e:
        logger.error(f"Error logging metrics for '{tool_name}': {e}", exc_info=True)

def get_tool_feedback(tool_name: str) -> str:
    """
    Retrieves and processes feedback for a specific tool from the metrics log file.

    Args:
        tool_name (str): The name of the tool.

    Returns:
        str: A text summary of the feedback, including success rates and errors. 
    """
    feedback_content: str = ""
    try:
        if not os.path.exists(METRICS_LOG_FILE):
            return f"Metrics log file '{METRICS_LOG_FILE}' not found."

        with open(METRICS_LOG_FILE, "r", encoding="utf-8") as f:
            all_metrics = [json.loads(line) for line in f if line.strip()]

        tool_metrics = [m for m in all_metrics if m["tool_name"] == tool_name]
        if not tool_metrics:
            return f"No metrics found for tool '{tool_name}'."

        total_executions: int = len(tool_metrics)
        successful_executions: int = sum(1 for m in tool_metrics if not m.get("error"))
        error_rate: float = (
            (1 - successful_executions / total_executions) * 100
            if total_executions > 0
            else 0
        )

        feedback_content += f"Feedback for tool '{tool_name}':\n"
        feedback_content += f"Total executions: {total_executions}\n"
        feedback_content += f"Successful executions: {successful_executions}\n"
        feedback_content += f"Error rate: {error_rate:.2f}%\n"

        if error_rate > 0:
            error_messages = [m["error"] for m in tool_metrics if m.get("error")]
            feedback_content += "Common errors:\n"
            for error in set(error_messages):
                feedback_content += f"- {error}\n"

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from metrics log: {e}", exc_info=True)
        return f"Error reading metrics for tool '{tool_name}'."
    except Exception as e:
        logger.error(f"Unexpected error getting feedback for tool '{tool_name}': {e}", exc_info=True)
        return f"Error processing feedback for tool '{tool_name}'."

    return feedback_content

###############################################################################
#                               TOOL FUNCTIONS                                #
###############################################################################
disk_io = DiskIOManager()

@tool
def visit_webpage(url: str) -> str:
    """
    Visits a webpage at the given URL and returns its content as a markdown string.

    This function fetches the webpage, saves its HTML content to disk, and converts
    that content to markdown, returning the final markdown string. If an error occurs,
    returns an error message.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to markdown, or an error message if the request fails.
    """
    tool_name: str = "visit_webpage"
    start_time: float = time.time()
    log_prefix: str = f"{tool_name}(url='{url}')"
    logger.info(f"{log_prefix}: Attempting to visit webpage.")

    output: str = ""
    error_msg: Optional[str] = None
    cache_dir: str = "./cache"

    try:
        response: requests.Response = requests.get(url, timeout=10)
        response.raise_for_status()

        raw_content_path: str = os.path.join(
            cache_dir,
            f"{urlparse(url).netloc}_{hashlib.md5(url.encode()).hexdigest()[:8]}.html"
        )
        os.makedirs(cache_dir, exist_ok=True)
        disk_io.write_data_to_disk(raw_content_path, response.text)

        logger.debug(f"{log_prefix}: Raw content saved to disk: {raw_content_path}")

        markdown_chunks: List[str] = []
        for chunk in disk_io.read_data_from_disk_in_chunks(raw_content_path):
            markdown_chunk: str = markdownify(chunk).strip()
            markdown_chunks.append(markdown_chunk)

        markdown_content: str = "\n\n".join(markdown_chunks)
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        output = markdown_content
        logger.info(f"{log_prefix}: Successfully fetched and converted webpage content.")
    except RequestException as e:
        error_msg = f"Error fetching the webpage: {str(e)}"
        logger.error(f"{log_prefix}: RequestException - {e}")
        output = error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.critical(f"{log_prefix}: Unexpected error - {e}", exc_info=True)
        output = error_msg
    finally:
        end_time: float = time.time()
        log_tool_metrics(tool_name, start_time, end_time, {"url": url}, output, error_msg)
        return output

@tool
def execute_python_code(code_string: str) -> str:
    """
    Executes a string of Python code and returns the output.

    Args:
        code_string: The Python code to execute.

    Returns:
        The output of the executed code, or an error message if an error occurred.
    """
    tool_name: str = "execute_python_code"
    start_time: float = time.time()
    log_prefix: str = f"{tool_name}(code_string='{code_string[:50]}...')"
    logger.info(f"{log_prefix}: Attempting to execute Python code.")

    output: str = ""
    error_msg: Optional[str] = None
    try:
        process = subprocess.Popen(
            ["python", "-c", code_string],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=15)
        if stderr:
            error_msg = f"Error executing code:\n{stderr}"
            logger.error(f"{log_prefix}: Error executing code: {stderr}")
            output = error_msg
        else:
            output = stdout
            logger.info(f"{log_prefix}: Successfully executed code.")
    except subprocess.TimeoutExpired:
        error_msg = "Error: Code execution timed out."
        logger.error(f"{log_prefix}: Code execution timed out.")
        output = error_msg
    except Exception as e:
        error_msg = f"Error executing code: {e}"
        logger.error(f"{log_prefix}: Unexpected error executing code: {e}", exc_info=True)
        output = error_msg
    finally:
        end_time: float = time.time()
        log_tool_metrics(tool_name, start_time, end_time, {"code_string": code_string}, output, error_msg)
        return output

###############################################################################
#                         ADDITIONAL TOOLS                         #
###############################################################################
@tool
def summarize_context() -> str:
    """
    Summarizes recent logs and metrics using the agent's model and returns a short summary.

    Returns:
        A one-paragraph summary of recent activities.
    """
    tool_name: str = "summarize_context"
    start_time: float = time.time()
    log_prefix: str = f"{tool_name}()"
    logger.info(f"{log_prefix}: Attempting to summarize recent context.")
    error_msg: Optional[str] = None
    summary: str = ""

    try:
        # Retrieve recent log entries
        log_content: str = ""
        try:
            with open("app.log", "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                log_content = "".join(all_lines[-50:])  # Get last 50 lines as context
        except FileNotFoundError:
            logger.warning(f"{log_prefix}: Log file not found, proceeding without log context.")
            log_content = ""
        except Exception as e:
            logger.error(f"{log_prefix}: Error reading log file: {e}", exc_info=True)
            log_content = ""

        # Retrieve recent metrics
        metrics_content: str = ""
        try:
            if os.path.exists(METRICS_LOG_FILE):
                with open(METRICS_LOG_FILE, "r", encoding="utf-8") as f:
                    all_metrics = [line.strip() for line in f.readlines()][-10:] # Get last 10 metrics
                    metrics_content = "\n".join(all_metrics)
            else:
                logger.warning(f"{log_prefix}: Metrics file not found, proceeding without metrics context.")
        except Exception as e:
            logger.error(f"{log_prefix}: Error reading metrics file: {e}", exc_info=True)
            metrics_content = ""

        combined_context: str = f"Recent Logs:\n{log_content}\n\nRecent Metrics:\n{metrics_content}"

        if not combined_context.strip():
            summary = "No recent logs or metrics available to summarize."
            logger.info(f"{log_prefix}: No context to summarize.")
            return summary

        prompt: str = f"Please summarize the following recent activities:\n\n{combined_context}\n\nSummary:"
        try:
            summary = model.generate_response(prompt)
            logger.info(f"{log_prefix}: Successfully generated context summary.")
        except Exception as e:
            error_msg = f"Error generating summary: {e}"
            logger.error(f"{log_prefix}: {error_msg}", exc_info=True)
            summary = f"Error generating summary: {e}"

    finally:
        end_time: float = time.time()
        log_tool_metrics(
            tool_name, start_time, end_time, {"input_context_source": "recent logs and metrics"}, summary, error_msg
        )
        return summary

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=10
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

###############################################################################
#                            MULTI-AGENT SETUP                                #
###############################################################################
# Model & Agents
MODEL_ID: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model = HfApiModel(MODEL_ID)

web_agent = ToolCallingAgent(
    tools=[
        DuckDuckGoSearchTool(),
        visit_webpage
    ],
    model=model,
    system_prompt=(
        "You are an advanced Eidosian assistant capable of performing comprehensive web searches, "
        "visiting webpages, and retrieving detailed information. Your responses should be accurate, "
        "context-aware, and aligned with the Eidos system's principles of modularity, efficiency, and "
        "adaptability. Ensure all outputs are consistent with the internal rule set and knowledge graph, "
        "and prioritize tasks based on urgency, complexity, and system load.\n\n"
        "{{managed_agents_descriptions}}"
    ),
    planning_interval=5,
    max_steps=20
)

managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="search",
    description="You are an advanced Eidosian assistant capable of performing comprehensive web searches, visiting webpages, and retrieving detailed information. Your responses should be accurate, context-aware, and aligned with the Eidos system's principles of modularity, efficiency, and adaptability. Ensure all outputs are consistent with the internal rule set and knowledge graph, and prioritize tasks based on urgency, complexity, and system load.",
    additional_prompting="You are an advanced Eidosian assistant capable of performing comprehensive web searches, visiting webpages, and retrieving detailed information. Your responses should be accurate, context-aware, and aligned with the Eidos system's principles of modularity, efficiency, and adaptability. Ensure all outputs are consistent with the internal rule set and knowledge graph, and prioritize tasks based on urgency, complexity, and system load.",
    provide_run_summary=True,
    managed_agent_prompt="You are an advanced Eidosian assistant capable of performing comprehensive web searches, visiting webpages, and retrieving detailed information. Your responses should be accurate, context-aware, and aligned with the Eidos system's principles of modularity, efficiency, and adaptability. Ensure all outputs are consistent with the internal rule set and knowledge graph, and prioritize tasks based on urgency, complexity, and system load."
    )
try:
    # Retrieve pip list in a chunkwise manner with error handling and logging
    pip_list_output = subprocess.check_output(["pip", "list"], text=True)
    installed_lines = pip_list_output.strip().split("\n")
    # Skip header lines (Package, Version, etc.)
    header_index = 2
    chunk_size = 50
    imports = []

    if len(installed_lines) > header_index:
        package_lines = installed_lines[header_index:]
        for i in range(0, len(package_lines), chunk_size):
            chunk = package_lines[i : i + chunk_size]
            for line in chunk:
                parts = line.split()
                if parts:
                    imports.append(parts[0])
    else:
        logger.warning("No packages found via pip list (unexpected format).")

except subprocess.CalledProcessError as e:
    logger.error(f"Error retrieving pip list: {e}", exc_info=True)
    imports = []
except Exception as e:
    logger.error(f"Unexpected error processing pip list: {e}", exc_info=True)
    imports = []

manager_agent = CodeAgent(
    tools=[],  # You could add more tool references here if needed
    model=model,
    system_prompt="You have access to '{{authorized_imports}}' to use in your code. The additional descritptions of the other agents you work with are '{{managed_agents_descriptions}}'. You are an advanced Eidosian assistant capable of performing comprehensive web searches, visiting webpages, and retrieving detailed information. Your responses should be accurate, context-aware, and aligned with the Eidos system's principles of modularity, efficiency, and adaptability. Ensure all outputs are consistent with the internal rule set and knowledge graph, and prioritize tasks based on urgency, complexity, and system load.\n\n{{managed_agents_descriptions}}",
    grammar=None,  # type: Dict[str, str] | None
    additional_authorized_imports=imports + ["time", "numpy", "pandas", "os", "requests", "huggingface_hub", "subprocess"],
    planning_interval=10,
    use_e2b_executor=False
)

###############################################################################
#                              AGENT LOOP DEMO                                #
###############################################################################
def agent_loop() -> None:
    """
    Demonstrates an AI agent's autonomous operation, broken into sub-steps for clarity.
    """
    print("Agent is starting... Step 1: Searching the internet.")
    action = "search_internet"
    query = "latest AI trends"
    print(f"Step 1a: The agent is performing action='{action}' with query='{query}'")
    results: str = DuckDuckGoSearchTool()(query=query)  # Equivalent to calling web_agent
    print("Step 1b: Search results obtained.\n", results)

    print("Step 2: Gathering feedback on the 'search_internet' tool usage.")
    feedback: str = get_tool_feedback("search_internet")
    print("Feedback:\n", feedback)

    print("Step 3: Summarizing context for improved decision-making.")
    context_summary: str = summarize_context(feedback)
    print(context_summary)

    print("Step 4: Check if user wants to continue or supply input.")
    user_input: str = input("User input (or type 'continue'): ")
    if user_input.lower() != "continue":
        print(f"Agent responding to user input: {user_input}")
    else:
        print("Agent continuing operation...")

###############################################################################
#                               MAIN EXECUTION                                #
###############################################################################
if __name__ == "__main__":
    # Example usage of some tools:
    # Step A: Visit webpage
    test_webpage: str = visit_webpage("https://www.example.com")
    print("Visited 'https://www.example.com' =>", test_webpage[:300], "...")

    # Step B: Execute a sample Python code string
    code_result: str = execute_python_code("print('Hello from executed code!')")
    print("Code Result:", code_result)

    # Manually run an example search using the manager_agent
    print("\n--- Manager Agent Example ---")
    try:
        answer = manager_agent.run(
            "If LLM trainings continue to scale up at the current rhythm until 2030, "
            "what would be the electric power in GW required to power the biggest training runs by 2030? "
            "What does that correspond to, compared to some countries? "
            "Please provide a source for any number used."
        )
        print("\nFinal Answer from Manager Agent:\n", answer)
    except Exception as e:
        logger.error(f"Error running the multi-agent system: {e}", exc_info=True)

    # Start the agent loop (conceptual demonstration with sub-steps)
    # agent_loop() 
