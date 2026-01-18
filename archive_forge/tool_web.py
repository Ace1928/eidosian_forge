import os
import re
import time
import logging
import threading
from collections import deque
from typing import List, Dict, Any, Generator, Tuple

import psutil
import requests
import huggingface_hub #type: ignore
from huggingface_hub import login #type: ignore
from requests.exceptions import RequestException #type: ignore
from markdownify import markdownify #type: ignore
from accelerate import disk_offload #type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM #type: ignore

from smolagents import tool #type: ignore
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    ManagedAgent,
    DuckDuckGoSearchTool,
    TransformersModel,
    MultiStepAgent,
    TOOL_CALLING_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    MANAGED_AGENT_PROMPT
)
from tools import visit_webpage, wiki, web_search, image_generator, DuckDuckGoSearchTool, HFModelDownloadsTool, ImageGeneratorTool, translator, document_qa, final_answer

# Authentication - Handles Hugging Face token login
HF_TOKEN = os.environ.get('HF_TOKEN') # Get the Hugging Face token from environment variables
if not HF_TOKEN: # Check if the token is set
    logging.warning("HF_TOKEN environment variable not set. Functionality may be limited.") # Log a warning if not set
else:
    login(token=HF_TOKEN) # Login to Hugging Face if the token is set

# Logging Configuration - Sets up basic logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(threadName)s - %(message)s')

# Constants for System Configuration and Performance Tuning - Defines various constants for the application
DEFAULT_OFFLOAD_DIR = "./offload_dir"  # Directory for disk offloading
DEFAULT_MAX_STEPS = 50 # Maximum steps for agent execution
DEFAULT_MAX_ITERATIONS = 25 # Maximum iterations for feedback loop
DEFAULT_SCORE_THRESHOLD = 9 # Target score for feedback loop
RESOURCE_MONITOR_INTERVAL = 5  # Interval for resource monitoring in seconds
HISTORY_LENGTH = 5 # Length of conversation history to keep
SUMMARY_PROMPT = "Summarize the following user conversation history, highlighting key decisions, requests, and feedback. Focus on maintaining crucial context for future interactions. Be extremely concise {{managed_agents_descriptions}}." # Prompt for summarizing user history
AI_SUMMARY_PROMPT = "Summarize the following AI output history, highlighting key actions, results, and any self-corrections or insights. Focus on maintaining crucial context for future actions. Be extremely concise {{managed_agents_descriptions}}." # Prompt for summarizing AI history
EXTRACT_KEY_POINTS_PROMPT = "Identify and extract the most critical information, requests, and feedback from the following conversation history. This information should be permanently retained as key memories. Be extremely concise {{managed_agents_descriptions}}." # Prompt for extracting key points from history

# Model Configuration - Defines the model path and ID
LOCAL_MODEL_DIR = "/home/lloyd/Development/saved_models/Qwen/Qwen2.5-0.5B-Instruct"  # Local absolute path to the model
MODEL_ID = os.environ.get("MODEL_ID", LOCAL_MODEL_DIR) # Get the model ID from environment variables or use the local path as default
logging.info(f"Using model from local path: {MODEL_ID}") # Log the model ID being used

# Model Initialization - Loads the model and tokenizer
tokenizer = None # Initialize tokenizer variable
model = None # Initialize model variable
try:
    # Load tokenizer and model from the specified MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) # Load the tokenizer
    model = disk_offload(AutoModelForCausalLM.from_pretrained(MODEL_ID), offload_dir=DEFAULT_OFFLOAD_DIR, offload_buffers=True) # Load the model with disk offloading
    logging.info(f"Model loaded successfully from '{MODEL_ID}'.") # Log successful model loading
    
    # Check if the model and tokenizer have already been saved to the local directory
    if not (os.path.exists(os.path.join(LOCAL_MODEL_DIR, "pytorch_model.bin")) and os.path.exists(os.path.join(LOCAL_MODEL_DIR, "tokenizer_config.json"))):
        # Save the model and tokenizer back to the local directory after loading
        tokenizer.save_pretrained(LOCAL_MODEL_DIR) # Save the tokenizer
        model.save_pretrained(LOCAL_MODEL_DIR, safe_serialization=False) # Save the model
        logging.info(f"Model and tokenizer saved to '{LOCAL_MODEL_DIR}'.") # Log successful save
    else:
        logging.info(f"Model and tokenizer already exist at '{LOCAL_MODEL_DIR}'. Skipping save.") # Log if model and tokenizer already exist
except Exception as e:
    logging.error(f"Failed to load model or tokenizer from '{MODEL_ID}': {e}", exc_info=True) # Log error if loading fails
    raise # Raise the exception to halt execution

class ResourceMonitor(threading.Thread):
    """
    Monitors CPU and memory usage for performance optimization and debugging.

    This thread runs in the background and periodically checks the system's CPU and memory usage.
    The collected data is stored in deques for calculating average usage.
    """
    def __init__(self, interval: int = RESOURCE_MONITOR_INTERVAL):
        """Initializes the ResourceMonitor with a specified interval."""
        super().__init__(daemon=True) # Initialize as a daemon thread
        self.interval = interval # Set the monitoring interval
        self._stop_event = threading.Event() # Event to stop the monitoring thread
        self.cpu_history: deque[float] = deque(maxlen=60) # Deque to store CPU usage history
        self.memory_history: deque[float] = deque(maxlen=60) # Deque to store memory usage history
        logging.debug(f"ResourceMonitor initialized with interval: {interval} seconds.") # Log initialization

    def run(self):
        """Runs the resource monitoring loop."""
        logging.debug("ResourceMonitor started.") # Log start of monitoring
        while not self._stop_event.is_set(): # Loop until stop event is set
            try:
                cpu_usage = psutil.cpu_percent() # Get current CPU usage
                memory_usage = psutil.virtual_memory().percent # Get current memory usage
                self.cpu_history.append(cpu_usage) # Add CPU usage to history
                self.memory_history.append(memory_usage) # Add memory usage to history
                logging.debug(f"Resource Usage - CPU: {cpu_usage:.2f}%, Memory: {memory_usage:.2f}%") # Log current resource usage
                time.sleep(self.interval) # Wait for the specified interval
            except Exception as e:
                logging.error(f"Error in resource monitoring: {e}", exc_info=True) # Log error if monitoring fails
                time.sleep(self.interval * 2)  # Reduced backoff - Wait longer if error occurs

    def stop(self):
        """Sets the stop event to terminate the monitoring loop."""
        logging.debug("ResourceMonitor stopping.") # Log stop of monitoring
        self._stop_event.set() # Set the stop event

    def get_average_usage(self) -> Tuple[float, float]:
        """Calculates and returns the average CPU and memory usage."""
        cpu_avg = sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0 # Calculate average CPU usage
        memory_avg = sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0 # Calculate average memory usage
        logging.debug(f"Average Resource Usage - CPU: {cpu_avg:.2f}%, Memory: {memory_avg:.2f}%") # Log average resource usage
        return cpu_avg, memory_avg # Return average CPU and memory usage

class AgentMemory:
    """
    Stores and manages conversation history, summaries, and permanent memories with thread safety.

    This class provides methods to add user and AI messages, retrieve recent history,
    manage summaries, and store permanent memories. Thread locks are used to ensure
    safe access in a multithreaded environment.
    """
    def __init__(self, history_length: int = HISTORY_LENGTH):
        """Initializes AgentMemory with a specified history length."""
        self.user_history: deque[str] = deque(maxlen=history_length) # Deque to store user message history
        self.ai_history: deque[str] = deque(maxlen=history_length) # Deque to store AI message history
        self.permanent_user_memory: List[str] = [] # List to store permanent user memories
        self.permanent_ai_memory: List[str] = [] # List to store permanent AI memories
        self.user_summary: str = "" # String to store user summary
        self.ai_summary: str = "" # String to store AI summary
        self.summary_lock = threading.Lock() # Lock for thread-safe access to summaries and memories
        logging.info(f"AgentMemory initialized with history length: {history_length}.") # Log initialization

    def add_user_message(self, message: str):
        """Adds a user message to the history."""
        with self.summary_lock: # Acquire lock for thread safety
            self.user_history.append(message) # Add message to user history
            logging.debug(f"Added user message: {message[:100]}...") # Log added user message

    def add_ai_message(self, message: str):
        """Adds an AI message to the history."""
        with self.summary_lock: # Acquire lock for thread safety
            self.ai_history.append(message) # Add message to AI history
            logging.debug(f"Added AI message: {message[:100]}...") # Log added AI message

    def get_recent_history(self) -> Tuple[List[str], List[str]]:
        """Returns the recent user and AI message history."""
        with self.summary_lock: # Acquire lock for thread safety
            return list(self.user_history), list(self.ai_history) # Return copies of user and AI history

    def get_summarized_history(self) -> Tuple[str, str]:
        """Returns the summarized user and AI history."""
        with self.summary_lock: # Acquire lock for thread safety
            return self.user_summary, self.ai_summary # Return user and AI summaries

    def update_user_summary(self, summary: str):
        """Updates the user summary."""
        with self.summary_lock: # Acquire lock for thread safety
            self.user_summary = summary # Update user summary
        logging.debug(f"Updated user summary: {summary[:100]}...") # Log updated user summary

    def update_ai_summary(self, summary: str):
        """Updates the AI summary."""
        with self.summary_lock: # Acquire lock for thread safety
            self.ai_summary = summary # Update AI summary
        logging.debug(f"Updated AI summary: {summary[:100]}...") # Log updated AI summary

    def add_permanent_user_memory(self, info: str):
        """Adds information to permanent user memory."""
        with self.summary_lock: # Acquire lock for thread safety
            if info not in self.permanent_user_memory: # Check if info already exists
                self.permanent_user_memory.append(info) # Add info to permanent user memory
                logging.info(f"Added to permanent user memory: {info}") # Log added permanent user memory

    def add_permanent_ai_memory(self, info: str):
        """Adds information to permanent AI memory."""
        with self.summary_lock: # Acquire lock for thread safety
            if info not in self.permanent_ai_memory: # Check if info already exists
                self.permanent_ai_memory.append(info) # Add info to permanent AI memory
                logging.info(f"Added to permanent AI memory: {info}") # Log added permanent AI memory

    def get_permanent_memory(self) -> Tuple[List[str], List[str]]:
        """Returns the permanent user and AI memories."""
        with self.summary_lock: # Acquire lock for thread safety
            return self.permanent_user_memory, self.permanent_ai_memory # Return permanent user and AI memories

# Initialize and start the resource monitor
resource_monitor = ResourceMonitor() # Initialize resource monitor
resource_monitor.start() # Start resource monitor thread

# Initialize Agent Memory
agent_memory = AgentMemory() # Initialize agent memory

# Initialize Core Tools - List of tools available to the agents
tools = [
    visit_webpage,
    wiki,
    web_search,
    image_generator,
    translator,
    document_qa,
    DuckDuckGoSearchTool(),
    HFModelDownloadsTool(),
    ImageGeneratorTool(),
    final_answer
]


# Instantiate the main model
code_qwen = None # Initialize main model variable
try:
    code_qwen = TransformersModel(MODEL_ID) # Load the main model
    logging.info(f"Main model '{MODEL_ID}' loaded.") # Log successful main model loading
except Exception as e:
    logging.critical("Failed to load main model.", exc_info=True) # Log critical error if loading fails
    raise # Raise the exception to halt execution

# Define the chat template for feedback and scoring - Template for feedback agent
chat_template = [
    {"role": "system", "content": "You are Eidos, a hyper-efficient AI. Evaluate model outputs against user queries with a score out of 10."}, # System prompt
    {"role": "user", "content": "Analyze this output relative to the original query. Provide a score out of 10 (10 is perfect). Be extremely critical and identify areas for improvement."}, # User prompt
    {"role": "assistant", "content": "Analysis of output vs. query: {output}. Score: {score}/10. Feedback: "}, # Assistant prompt
]
memory_model = code_qwen
code_qwen = code_qwen
model= code_qwen

# Define Feedback Agent - Agent for providing feedback on other agent's outputs
feedback_agent = ToolCallingAgent(
    tools=tools, # Tools available to the feedback agent
    model=code_qwen, # Model used by the feedback agent
    max_steps=4, # Maximum steps for feedback agent
    add_base_tools=True, # Add base tools
    verbosity_level=5, # Set verbosity level
    planning_interval=2, # Set planning interval
    system_prompt=TOOL_CALLING_SYSTEM_PROMPT
)

# Define summarizer agents - Agents for summarizing user and AI history
user_history_summarizer = ToolCallingAgent(
    tools=tools, # No tools for summarizer agents
    model=memory_model, # Model used by user history summarizer
    planning_interval=2,
    add_base_tools=True,
    verbosity_level=5,
    system_prompt=TOOL_CALLING_SYSTEM_PROMPT
)

ai_history_summarizer = ToolCallingAgent(
    tools=tools, # No tools for summarizer agents
    model=memory_model, # Model used by AI history summarizer
    planning_interval=2,
    add_base_tools=True,
    verbosity_level=5,
)

# Initialize the Web Agent with all tools - Agent for web-based tasks
web_agent = ToolCallingAgent(
    tools=tools,  # All tools available to the web agent
    model=code_qwen, # Model used by the web agent
    max_steps=10, # Maximum steps for web agent
    planning_interval=2,
    add_base_tools=True,
    verbosity_level=5,
    system_prompt=TOOL_CALLING_SYSTEM_PROMPT
)

# Define the memory manager agent - Agent for managing memory and summarization
memory_manager_agent = CodeAgent(
    tools=tools, # No tools for memory manager agent
    model=memory_model, # Model used by memory manager agent
    managed_agents=[ # List of managed agents
        ManagedAgent(agent=user_history_summarizer, name="user_summarizer", description="Summarizes user history."), # User history summarizer
        ManagedAgent(agent=ai_history_summarizer, name="ai_summarizer", description="Summarizes AI history.") # AI history summarizer
    ],
    planning_interval=2,
    add_base_tools=True,
    verbosity_level=5,
    additional_authorized_imports=["time", "numpy", "pandas"],
    system_prompt=CODE_SYSTEM_PROMPT
)

# Define the managed web agent - Managed agent for web-based tasks
managed_web_agent = ManagedAgent(
    agent=web_agent, # The web agent
    name="search", # Name of the managed agent
    description="Runs web searches for you. Give it your query as an argument." + MANAGED_AGENT_PROMPT, # Description of the managed agent
)

managed_ai_history_agent = ManagedAgent(
    agent=ai_history_summarizer,
    name="ai_history",
    description="Manages AI history and summarization." + MANAGED_AGENT_PROMPT,
)

managed_user_history_agent = ManagedAgent(
    agent=user_history_summarizer,
    name="user_history",
    description="Manages user history and summarization." + MANAGED_AGENT_PROMPT,
)

managed_memory_agent = ManagedAgent(
    agent=memory_manager_agent,
    name="memory",
    description="Manages memory and summarization." + MANAGED_AGENT_PROMPT,
)

managed_feedback_agent = ManagedAgent(
    agent=feedback_agent,
    name="feedback",
    description="Provides feedback on the output of other agents." + MANAGED_AGENT_PROMPT,
)

# Define the main manager agent - Main agent that manages other agents
manager_agent = CodeAgent(
    tools=[visit_webpage,
    wiki,
    web_search,
    image_generator,
    translator,
    document_qa,
    DuckDuckGoSearchTool(),
    HFModelDownloadsTool(),
    ImageGeneratorTool(),
    final_answer], # No tools for main manager agent
    model=code_qwen, # Model used by main manager agent
    managed_agents=[managed_web_agent, managed_ai_history_agent, managed_user_history_agent, managed_memory_agent, managed_feedback_agent], # List of managed agents
    additional_authorized_imports=["time", "numpy", "pandas","matplotlib","pygame","requests"], # Additional authorized imports
    planning_interval=1,
    add_base_tools=True,
    system_prompt=CODE_SYSTEM_PROMPT
)

def summarize_history(agent: ToolCallingAgent, history: str, prompt: str) -> str:
    """
    Summarizes a given history using the specified agent.

    Args:
        agent: The agent to use for summarization.
        history: The string representing the history to summarize.
        prompt: The prompt to guide the summarization.

    Returns:
        The summarized history.

    Raises:
        ValueError: If the agent is invalid.
        Exception: If an error occurs during the summarization process.
    """
    agent_name = type(agent).__name__ # Get the name of the agent
    agent_count = 0 # Initialize agent count
    for existing_agent in memory_manager_agent.managed_agents: # Loop through managed agents
        if hasattr(existing_agent, 'agent') and existing_agent.agent == agent: # Check if the agent is the same
            agent_count += 1 # Increment agent count
    agent_name = f"{agent_name}_{agent_count}" # Create a unique agent name
    logging.debug(f"Attempting to summarize history using agent: {agent_name}") # Log attempt to summarize
    if not history: # Check if history is empty
        logging.debug(f"No history to summarize for {agent_name}.") # Log if no history
        return "" # Return empty string if no history
    try:
        response_generator = manager_agent.run( # Run the agent
            prompt.format(managed_agents_descriptions=""), # Format the prompt
            stream=False, # Disable streaming
            additional_args={"history": history} # Pass history as additional argument
        )
        if response_generator is None: # Check if response generator is None
            logging.error(f"Response generator is None for {agent_name}.") # Log error if response generator is None
            return "" # Return empty string if response generator is None
        response = "".join(str(token) for token in response_generator) # Join the tokens into a string
        logging.info(f"{agent_name} successfully summarized history.") # Log successful summarization
        return response.strip() # Return the summarized history
    except Exception as e:
        logging.error(f"Error during history summarization with {agent_name}: {e}", exc_info=True) # Log error if summarization fails
        return "" # Return empty string if summarization fails

def extract_key_info(agent: ToolCallingAgent, history: str, prompt: str) -> List[str]:
    """
    Extracts key information from the history using the specified agent.

    Args:
        agent: The agent to use for extraction.
        history: The string representing the history to extract from.
        prompt: The prompt to guide the key information extraction.

    Returns:
        A list of key information points extracted from the history.

    Raises:
        ValueError: If the agent is invalid.
        Exception: If an error occurs during the extraction process.
    """
    agent_name = type(agent).__name__ # Get the name of the agent
    agent_count = 0 # Initialize agent count
    for existing_agent in memory_manager_agent.managed_agents: # Loop through managed agents
        if hasattr(existing_agent, 'agent') and existing_agent.agent == agent: # Check if the agent is the same
            agent_count += 1 # Increment agent count
    agent_name = f"{agent_name}_{agent_count}" # Create a unique agent name
    logging.debug(f"Attempting to extract key info from history using agent: {agent_name}") # Log attempt to extract key info
    if not history: # Check if history is empty
        logging.debug(f"No history to extract key info from for {agent_name}.") # Log if no history
        return [] # Return empty list if no history
    try:
        response_generator = manager_agent.run(prompt.format(managed_agents_descriptions=""), stream=False, additional_args={"history": history}) # Run the agent
        if response_generator is None: # Check if response generator is None
            logging.error(f"Response generator is None for {agent_name}.") # Log error if response generator is None
            return [] # Return empty list if response generator is None
        response = "".join(token for token in response_generator if isinstance(token, str)) # Join the tokens into a string
        key_points = [item.strip() for item in response.strip().split('\n') if item.strip()] # Split the response into key points
        logging.info(f"{agent_name} successfully extracted key information.") # Log successful extraction
        return key_points # Return the list of key points
    except Exception as e:
        logging.error(f"Error during key information extraction with {agent_name}: {e}", exc_info=True) # Log error if extraction fails
        return [] # Return empty list if extraction fails

def run_agent_with_feedback(
    agent: MultiStepAgent,
    query: str,
    chat_template: List[Dict[str, str]],
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    score_threshold: int = DEFAULT_SCORE_THRESHOLD,
    offload_dir: str = DEFAULT_OFFLOAD_DIR,
) -> str:
    """
    Runs an agent, incorporates feedback, and iterates until a satisfactory score is achieved.

    [Docstring truncated for brevity...]
    """
    output = ""
    score = 0
    iteration = 0
    previous_output = query
    stop_event = threading.Event()
    logging.info(f"Starting agent run with query: '{query[:100]}...', max_iterations: {max_iterations}, score_threshold: {score_threshold}.")

    agent_name = f"{type(agent).__name__}_0"  # Simplified agent naming

    def prepare_context() -> str:
        """Assembles context with memories and recent histories."""
        recent_user_history, recent_ai_history = agent_memory.get_recent_history()
        user_summary, ai_summary = agent_memory.get_summarized_history()
        permanent_user_memory, permanent_ai_memory = agent_memory.get_permanent_memory()
        context = (
            f"Permanent User Memories: {permanent_user_memory}\n"
            f"Permanent AI Memories: {permanent_ai_memory}\n"
            f"User Summary: {user_summary}\n"
            f"AI Summary: {ai_summary}\n"
            f"Recent User Messages: {recent_user_history}\n"
            f"Recent AI Messages: {recent_ai_history}"
        )
        return f"{context}\n\nUser Query: {previous_output}"

    def check_resources():
        cpu_avg, mem_avg = resource_monitor.get_average_usage()
        if cpu_avg > 90 or mem_avg > 90:
            logging.warning(f"System resources high (CPU: {cpu_avg:.2f}%, Memory: {mem_avg:.2f}%).")

    def handle_feedback(output: str, query: str) -> int:
        """Sends feedback to the feedback_agent and extracts a score."""
        feedback_chat = [
            {"role": "system", "content": chat_template[0]["content"]},
            {"role": "user", "content": chat_template[1]["content"]},
            {"role": "assistant", "content": chat_template[2]["content"].format(query=query, output=output, score=score)},
        ]
        try:
            feedback_gen = manager_agent.run(
                str(feedback_chat),
                stream=False,
                additional_args={"history": agent_memory.get_recent_history()}
            )
            if feedback_gen is None:
                logging.error("Feedback response generator is None.")
                return 0
            feedback_response = "".join(token for token in feedback_gen if isinstance(token, str))
        except Exception as e:
            logging.error("Feedback agent execution error.", exc_info=True)
            return 0

        score_match = re.search(r"Score: (\d+)/10", feedback_response)
        if score_match:
            try:
                return int(score_match.group(1))
            except ValueError:
                logging.warning(f"Could not parse integer score from feedback response: {feedback_response}")
        else:
            logging.warning(f"Could not extract score from feedback response: {feedback_response}")
        return 0

    def agent_runner(current_input: str) -> str:
        """Runs the agent and returns the full response."""
        augmented_input = (
            f"{current_input}\n\n"
            "Instructions:\n"
            "1. Use the web to find information and cite your sources.\n"
            "2. Perform calculations using your python environment.\n"
            "3. Include detailed logging and error handling in your code.\n"
            "4. When you believe the task is complete and the answer is satisfactory, state your confidence as 'Score: {score}/10' where {score} is your confidence (9 or greater for completion).\n"
            "5. If you are unsure or the task is not complete, state 'Score: 0/10'."
        )
        logging.debug(f"Agent executing with input: '{augmented_input[:200]}...'")
        try:
            response_gen = manager_agent.run(
                augmented_input,
                stream=False,
                additional_args={"history": agent_memory.get_recent_history()}
            )
            if response_gen is None:
                logging.error(f"Response generator is None for {agent_name}.")
                return ""
            return "".join(token for token in response_gen if isinstance(token, str))
        except Exception as e:
            logging.error("Agent execution error.", exc_info=True)
            return ""

    while score < score_threshold and iteration < max_iterations and not stop_event.is_set():
        logging.info(f"Iteration {iteration + 1}/{max_iterations}, current score: {score}/{score_threshold}.")
        try:
            check_resources()
            current_input = prepare_context()
            agent_memory.add_user_message(previous_output)

            response_text = agent_runner(current_input)
            if not response_text:
                logging.warning("Agent produced no output.")
                break

            # Process agent output for termination condition based on score
            score_match_iter = re.search(r"Score: (\d+)/10", response_text)
            if score_match_iter:
                try:
                    potential_score = int(score_match_iter.group(1))
                    if potential_score >= score_threshold:
                        logging.info(f"Agent achieved target score ({potential_score}), terminating early.")
                        score = potential_score
                        stop_event.set()
                except ValueError:
                    logging.warning(f"Could not parse score from agent output: {response_text}")

            output = response_text.strip()
            agent_memory.add_ai_message(output)

            # Summarization and key info extraction with detailed error handling
            try:
                user_summary_text = summarize_history(user_history_summarizer, "\n".join(agent_memory.user_history), SUMMARY_PROMPT)
                agent_memory.update_user_summary(user_summary_text)
            except Exception as e:
                logging.error(f"Error during user history summarization: {e}", exc_info=True)

            try:
                ai_summary_text = summarize_history(ai_history_summarizer, "\n".join(agent_memory.ai_history), AI_SUMMARY_PROMPT)
                agent_memory.update_ai_summary(ai_summary_text)
            except Exception as e:
                logging.error(f"Error during AI history summarization: {e}", exc_info=True)

            try:
                key_user_info = extract_key_info(user_history_summarizer, "\n".join(agent_memory.user_history), EXTRACT_KEY_POINTS_PROMPT)
                for info in key_user_info:
                    agent_memory.add_permanent_user_memory(info)
            except Exception as e:
                logging.error(f"Error during user key info extraction: {e}", exc_info=True)

            try:
                key_ai_info = extract_key_info(ai_history_summarizer, "\n".join(agent_memory.ai_history), EXTRACT_KEY_POINTS_PROMPT)
                for info in key_ai_info:
                    agent_memory.add_permanent_ai_memory(info)
            except Exception as e:
                logging.error(f"Error during AI key info extraction: {e}", exc_info=True)

            # Handle feedback to update score
            score = handle_feedback(output, query)
            logging.info(f"Current score: {score}")

            previous_output = output
            iteration += 1

        except Exception as e:
            logging.error(f"An error occurred during agent execution: {e}", exc_info=True)
            break

    if score >= score_threshold:
        logging.info(f"Achieved target score of {score} after {iteration} iterations.")
    else:
        logging.warning(f"Did not achieve target score of {score_threshold} after {iteration} iterations.")
    return output

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
        import pygame
        for i in range(20):
            prompt = manager_agent.run(task="20 word or less detailed output of a random scene/image", stream=True, single_step=True)
            final_prompt = final_answer(prompt)
            try:
                image_path = image_generator(final_prompt)
                print(f"Image {i+1} generated with image_generator at: {image_path}")
            except Exception as e:
                print(f"Error generating image: {e}")
                break
            
            # Display the image using pygame
            pygame.init()
            if isinstance(image_path, str) and os.path.exists(image_path):
                image = pygame.image.load(image_path)
                screen = pygame.display.set_mode(image.get_size())
                screen.blit(image, (0, 0))
                pygame.display.flip()
            else:
                print(f"Failed to load image from {image_path}")
                break

            # Wait for the user to close the window or press a key
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                        waiting = False
            pygame.quit()

    except Exception as e:
        print(f"Error testing image_generator: {e}")

    # Test ImageGeneratorTool directly
    try:
        prompt = "A cat wearing a hat"
        final_prompt = final_answer(prompt)
        image_path = ImageGeneratorTool()(final_prompt)
        print(f"Image generated with ImageGeneratorTool at: {image_path}")
    except Exception as e:
        print(f"Error testing ImageGeneratorTool: {e}")
    
    try:
        prompt = manager_agent.run(task="Who are you? Output in 20 words or less", single_step=True)
        print(f"Prompt: {prompt}")
        manager_agent.run(task=f"Make a nice detailed prompt, 50 words or less, then generate an image of it using your image generator tool", stream=False)
    except Exception as e:
        print(f"Error testing image generation agent: {e}")

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

    query = """
    If LLM training continues to scale up at the current rhythm until 2030, what would be the electric power in GW required to power the biggest training runs by 2030? What would that correspond to, compared to some countries? Please provide a source for any numbers used.
    """ # Example query

    output = manager_agent.run(task=query, stream=True)
    print(f"Output: {output}")
    final_answer_output = final_answer(str(output))
    print(f"Final answer output: {final_answer_output}")

    # Ensure resource monitor is stopped properly
    if resource_monitor.is_alive(): # Check if resource monitor is alive
        resource_monitor.stop() # Stop the resource monitor
        resource_monitor.join() # Join the resource monitor thread
        logging.info("Resource monitor stopped.") # Log that the resource monitor has stopped
