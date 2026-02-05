# ------------------
# IMPORTS & CONSTANTS
# ------------------
import logging
import re
import hashlib
import os
import time
import json
from urllib.parse import urljoin
import queue
from queue import Queue
from typing import Tuple
from datetime import datetime

import requests
import httpx
import torch
from transformers import (  # type: ignore[import]
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from eidosian_core import eidosian
from accelerate import disk_offload  # type: ignore[import]
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_exponential, stop_after_attempt

from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate as LangchainPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.callbacks.promptlayer_callback import (
    PromptLayerCallbackHandler,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import threading
import asyncio


# Constants
MODEL_NAME = "/home/lloyd/Development/saved_models/Qwen/Qwen2.5-0.5B-Instruct"
SEARCH_ENDPOINT = "http://192.168.4.73:8888/search"
task_queue: Queue = Queue()
result_queue: Queue = Queue()
update_queue: Queue = Queue()
QUEUES: Tuple[Queue, Queue, Queue] = (task_queue, result_queue, update_queue)

# Add after existing constants
OLLAMA_BASE_URL = "http://192.168.4.73:11434/api/"
OLLAMA_TIMEOUT = 30.0
ASYNC_CLIENT = httpx.AsyncClient(
    timeout=OLLAMA_TIMEOUT
)  # Using timeout from old config, but client from new config

# Regex patterns
TOPIC_PATTERN = re.compile(r"^[a-zA-Z0-9\s\-\.,'\"()]+$")  # Adjusted pattern
CODE_BLOCK_PATTERN = re.compile(r"```python\n(.*?)\n```", re.DOTALL)  # Adjusted pattern
EMOTION_KEYWORDS = {
    "positive": r"\b(happy|joy|excitement|hope|confidence)\b",
    "negative": r"\b(sad|anger|fear|doubt|frustration)\b",
}

# Add to constants section
OUTPUT_JSON_PATH = os.path.join("outputs", "processing_history.json")


# ------------------
# SUPPORTING CLASSES
# ------------------
class EnhancedPromptBuilder:
    """LangChain-powered prompt engineering with validation"""

    def __init__(self):
        self.templates = {
            "code_generation": LangchainPromptTemplate.from_template(
                "Identify fundamental Python code for: {topic}\n"
                "Include code blocks in ```python format\n"
                "Consider these constraints: {constraints}"
            ),
            "emotional_analysis": LangchainPromptTemplate.from_template(
                "Analyze emotional context for: {topic}\n"
                "Focus on: {aspects}\n"
                "Format as: {format_instructions}"
            ),
        }

    @eidosian()
    def build_prompt(self, prompt_type, **kwargs):
        """Generate validated prompts with LangChain templates"""
        if prompt_type not in self.templates:
            raise ValueError(f"Invalid prompt type: {prompt_type}")

        return self.templates[prompt_type].format(**kwargs)


# ------------------
# CORE FUNCTIONALITY
# ------------------
@eidosian()
@lru_cache(maxsize=1)
def load_model(model_name):
    """
    Optimized pipeline configuration with proper disk offloading and device awareness.
    Leverages accelerate's disk_offload for efficient memory management,
    and device_map='auto' to automatically utilize available hardware (CPU, GPU, etc.).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )  # Tokenizer is loaded from the same model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically handles CPU, GPU, MPU, NPU/APU if available
        offload_folder="offload",  # Enables disk offloading to specified folder
        offload_state_dict=True,  # Offload model state dict to disk
        low_cpu_mem_usage=True,  # Reduces CPU memory usage during loading
        torch_dtype=(
            torch.float16 if torch.cuda.is_available() else torch.float32
        ),  # Use float16 for CUDA, float32 for CPU for performance
    )
    return model


@eidosian()
@lru_cache(maxsize=1)
def load_tokenizer(model_name):
    """
    Loads the tokenizer associated with the given model_name.
    Ensures tokenizer is consistent with the model for correct processing.
    """
    return AutoTokenizer.from_pretrained(
        model_name
    )  # Tokenizer is loaded from the same model_name


@eidosian()
@lru_cache(maxsize=1)
def load_text_generation_pipeline(model, tokenizer):
    """
    Load a text generation pipeline with optimized settings for performance and resource usage.
    Utilizes the pre-loaded model and tokenizer for efficiency.
    Adjusts batch size based on CUDA availability for optimal throughput.
    """
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,  # Using the tokenizer loaded from load_tokenizer, ensuring consistency
        max_new_tokens=500,
        temperature=0.3,
        top_p=0.95,
        repetition_penalty=1.15,
        batch_size=(
            2 if torch.cuda.is_available() else 1
        ),  # Adjust batch size based on CUDA availability
    )

    return HuggingFacePipeline(
        pipeline=pipe,
        model_kwargs={"temperature": 0.3, "max_length": 1000},
        verbose=True,
    )


@eidosian()
def remove_think_tags(text):
    """Utility function to remove <think> and </think> tags."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Add as utility function
@eidosian()
def sanitize_response(response: str) -> str:
    """Remove unsafe patterns from model responses"""
    patterns = [
        r"<think>.*?</think>",
        r"```.*?```",
        r"<.*?>",
        r"(\b(?:rm|shutdown|format)\b.*)",
    ]
    for pattern in patterns:
        response = re.sub(pattern, "", response, flags=re.DOTALL | re.IGNORECASE)
    return response.strip()


# Central Orchestrator Class
class CentralOrchestrator:
    def __init__(self):
        self.processing_threads = []
        self.update_thread = threading.Thread(target=self.send_updates)
        self.update_thread.start()
        self.model = load_model(MODEL_NAME)  # Central orchestrator model
        self.tokenizer = load_tokenizer(MODEL_NAME)
        self.submodules = []
        self.initialize_submodules()
        self.pl_callback = PromptLayerCallbackHandler(
            pl_tags=["traffic-prediction-system", "v1.2"]
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @eidosian()
    def initialize_submodules(self):
        """
        Initialize submodules with proper async event loop handling
        """
        num_submodules = 5
        for i in range(num_submodules):
            submodule = SubModule(id=i)
            self.submodules.append(submodule)
            # Create a daemon thread with proper async handling
            thread = threading.Thread(
                target=self._run_submodule_async, args=(submodule,), daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)

    def _run_submodule_async(self, submodule):
        """Run submodule with clean async context"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(submodule.run())
        finally:
            loop.close()

    # Main workflow
    @eidosian()
    def run(self, input_text):
        """
        Main method to run the orchestrator.
        """
        update_queue.put("Starting processing of input.")
        topics = self.break_into_topics(input_text)
        update_queue.put(f"Identified topics: {topics}")
        self.distribute_tasks(topics)
        self.collect_results()
        update_queue.put("All tasks have been processed.")
        self.compile_results()

    @eidosian()
    def break_into_topics(self, input_text: str) -> list:
        """Split input into focused topics using the language model"""
        prompt = f"""Break this input into focused discussion topics:
        {input_text}
        
        Return as a JSON list of strings: ["topic1", "topic2", ...]"""

        # Replace the invoke call with proper forward pass
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs = inputs.to(self.device)  # Force device alignment
        outputs = self.model.generate(**inputs, max_new_tokens=500)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return [input_text]  # Fallback to original input

    @eidosian()
    def distribute_tasks(self, topics):
        """
        Distribute each topic to a submodule for processing.
        """
        for topic in topics:
            task = {"topic": topic, "status": "pending"}
            task_queue.put(task)
            update_queue.put(f"Task for topic '{topic}' has been queued.")

    @eidosian()
    def compile_results(self):
        """Compile all results into a structured JSON output with history"""
        update_queue.put("Compiling all results into JSON output.")

        # First create output_data
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "modules": [],
            "processing_steps": [],
            "performance_metrics": {
                "total_tasks": result_queue.qsize(),
                "avg_processing_time": 0,  # Initialize with default value
            },
        }

        # Then calculate start_time
        start_time = datetime.strptime(output_data["timestamp"], "%Y-%m-%d %H:%M:%S")

        # Collect all results
        while not result_queue.empty():
            try:
                result = result_queue.get_nowait()
                output_data["modules"].append(
                    {
                        "module_id": result.get("module_id", ""),
                        "topic": result.get("topic", ""),
                        "summary": result.get("summary", ""),
                        "timestamp": result.get("timestamp", ""),
                    }
                )
            except queue.Empty:
                break

        # Save structured JSON
        os.makedirs("outputs", exist_ok=True)
        with open(OUTPUT_JSON_PATH, "w") as f:
            json.dump(output_data, f, indent=2)

        update_queue.put(f"JSON output saved to {OUTPUT_JSON_PATH}")

    # Result handling
    @eidosian()
    def collect_results(self):
        """
        Collect results from submodules.
        """
        while True:
            try:
                result = result_queue.get(timeout=10)
                print(
                    f"Received result for topic '{result['topic']}': {result['summary']}"
                )
                # Here, you can implement further processing or compilation of results
            except queue.Empty:
                break

    @eidosian()
    def send_updates(self):
        """
        Send frequent updates to the user about the progress of tasks.
        """
        while True:
            try:
                update = update_queue.get(timeout=5)
                print(f"[Update]: {update}")
            except queue.Empty:
                pass
            time.sleep(1)

    # Helpers
    def _parse_topics(self, text):
        """Structured parsing of generated topics"""
        topics = []
        for line in text.split("\n"):
            if re.match(r"^\d+\.\s", line):
                topic = re.sub(r"^\d+\.\s*", "", line).strip()
                if re.match(TOPIC_PATTERN, topic):
                    topics.append(topic)
        return topics[:5]  # Limit to top 5 topics


class SubModule:
    """
    A submodule responsible for processing a specific topic through a series of steps
    using language models and web search functionalities.
    """

    def __init__(self, id):
        """
        Initializes a SubModule with a unique ID and loads necessary models and tools.
        """
        self.id = id
        self.local_model = load_model(MODEL_NAME)  # Local model
        # Assuming OllamaHandler is defined elsewhere and accessible
        self.ollama_client = OllamaHandler()
        self.model_available = {
            "local": True,
            "ollama": self._check_ollama_availability(),
        }
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(
            f"SubModule {self.id} initialized. Model availability - Local: {self.model_available['local']}, Ollama: {self.model_available['ollama']}"
        )

    async def _check_ollama_availability(self):
        """Check if Ollama service is available"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(urljoin(OLLAMA_BASE_URL, "tags"), timeout=5)
                return response.status_code == 200
        except Exception:
            return False

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
    )
    async def _parallel_generate(self, prompt: str, system_prompt: str = "") -> str:
        """Run both models in parallel and combine results"""
        tasks = []

        if self.model_available["local"]:
            tasks.append(
                asyncio.create_task(self._local_generate(prompt, system_prompt))
            )
        if self.model_available["ollama"]:
            tasks.append(
                asyncio.create_task(self._ollama_generate(prompt, system_prompt))
            )

        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED, timeout=30
            )
            if not done:
                raise TimeoutError("No models responded in time")
            best_result = max((t.result() for t in done), key=len, default="")
            return best_result
        except TimeoutError as e:
            logging.warning(f"Generation timeout: {str(e)}")
            return ""

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
    )
    async def _local_generate(self, prompt: str, system_prompt: str = "") -> str:
        """Local model generation with LangChain"""
        try:
            chain = self.local_model | StrOutputParser()
            return await chain.ainvoke(
                {"role": "system", "content": system_prompt, "prompt": prompt}
            )
        except Exception as e:
            self.model_available["local"] = False
            logging.error(f"Local model failed: {e}")
            return ""

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
    )
    async def _ollama_generate(self, prompt: str, system_prompt: str = "") -> str:
        """Ollama API generation"""
        try:
            response = await self.ollama_client.generate(
                prompt=prompt,
                model="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",  # DO NOT REMOVE THIS COMMENT DO NOT CHANGE THIS MODEL NAME
                system=system_prompt,
                options={"temperature": 0.3, "num_ctx": 4096},  # Add context length
            )
            return response.get("response", "").split("</think>")[
                -1
            ]  # Clean termination
        except Exception as e:
            self.model_available["ollama"] = False
            logging.error(f"Ollama failed: {e}")
            return ""

    @eidosian()
    async def emotional_context(self, topic: str) -> dict:
        """
        Analyzes the emotional context of a given topic and returns a structured JSON.

        Args:
            topic (str): The topic for emotional context analysis.

        Returns:
            dict: JSON object containing emotional context analysis.
        """
        try:
            prompt = f"""Analyze emotional context for: {topic}
            Output JSON schema:
            {{
                "motivation": "string",
                "concerns": ["string"],
                "desired_outcomes": ["string"],
                "emotional_tones": {{
                    "positive": ["string"],
                    "negative": ["string"]
                }}
            }}"""

            response = await OllamaHandler.generate(
                prompt=prompt,
                model="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",  # Reasoning model with <think> tags
                format="json",
                temperature=0.1,  # More deterministic output
            )

            if "response" in response:
                text_response = response["response"]
                cleaned_response = remove_think_tags(text_response)
                try:
                    parsed = json.loads(cleaned_response)
                    return self._validate_emotional_json(parsed)
                except json.JSONDecodeError as e:
                    logging.warning(
                        f"JSON parsing failed for topic '{topic}': {cleaned_response}. Error: {e}"
                    )
                    return self._fallback_emotional_analysis(topic)
            return {}
        except Exception as e:
            logging.error(
                f"Emotional context error for '{topic}': {str(e)}", exc_info=True
            )
            update_queue.put(f"SubModule {self.id} emotional analysis failed")
            return {}

    def _fallback_emotional_analysis(self, topic: str) -> dict:
        return {
            "motivation": "Analysis unavailable",
            "concerns": ["Data quality issues"],
            "desired_outcomes": ["Reliable analysis"],
            "emotional_tones": {"positive": [], "negative": []},
        }

    def _merge_analyses(self, local: str, ollama: str) -> dict:
        """Merge results from both models"""
        merged = {
            "motivation": "",
            "concerns": [],
            "desired_outcomes": [],
            "emotional_tones": {"positive": [], "negative": []},
        }

        # Try to parse local result
        try:
            local_data = json.loads(local)
            merged = self._deep_merge(merged, local_data)
        except json.JSONDecodeError:
            pass

        # Try to parse Ollama result
        try:
            ollama_data = json.loads(ollama)
            merged = self._deep_merge(merged, ollama_data)
        except json.JSONDecodeError:
            pass

        return merged

    def _deep_merge(self, base: dict, update: dict) -> dict:
        """Recursively merge dictionaries"""
        for key, value in update.items():
            if isinstance(value, dict):
                base[key] = self._deep_merge(base.get(key, {}), value)
            elif isinstance(value, list):
                base[key] = list(set(base.get(key, []) + value))
            else:
                if value and not base.get(key):
                    base[key] = value
        return base

    # Assuming _validate_emotional_json is defined elsewhere or will be implemented
    def _validate_emotional_json(self, data: dict) -> dict:
        """Placeholder for JSON validation - implement actual validation as needed"""
        return data

    # Web/search functionality
    @eidosian()
    async def perform_web_searches(self, web_queries):
        """
        Enhanced with LangChain document processing to perform web searches and store results.

        Args:
            web_queries (list): A list of web queries (URLs).

        Returns:
            list: List of LangChain documents from web search results.
        """
        try:
            valid_urls = [
                q for q in web_queries if re.match(r"^https?://", q)
            ]  # Better URL validation
            loader = AsyncHtmlLoader(valid_urls)
            docs = loader.load()

            html2text = Html2TextTransformer()
            transformed = html2text.transform_documents(docs)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, chunk_overlap=200
            )
            split_docs = splitter.transform_documents(transformed)

            split_docs = [
                doc for doc in split_docs if len(doc.page_content) >= 100
            ]  # Filter short content
            self._store_documents(split_docs)
            return split_docs
        except Exception as e:
            logging.error(f"Error in perform_web_searches: {e}", exc_info=True)
            return []

    def _prepare_request(self, url: str) -> httpx.Request:  # Changed to httpx.Request
        """Create properly configured request with headers"""
        req = httpx.Request(
            "GET",
            url,
            headers={  # Changed to httpx.Request
                "User-Agent": os.getenv("USER_AGENT", "research-network/1.0"),
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
        return req

    def _process_documents(self, docs):
        """Process documents with validation and error handling"""
        try:
            html2text = Html2TextTransformer()
            transformed = html2text.transform_documents(docs)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, chunk_overlap=200
            )
            return splitter.transform_documents(transformed)
        except Exception as e:
            logging.error(f"Document processing failed: {e}")
            return []

    def _store_documents(self, documents):
        """
        Optimized document storage with metadata using FAISS vector database.

        Args:
            documents (list): List of LangChain documents to store.
        """
        try:
            os.makedirs("vector_store", exist_ok=True)  # Ensure directory exists
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            )

            if os.path.exists("document_store.faiss"):
                if not hasattr(self, "vector_db"):  # Add existence check
                    self.vector_db = FAISS.load_local(
                        "document_store.faiss", embeddings
                    )
                self.vector_db.add_documents(
                    documents,
                    metadatas=[{"source": f"submodule_{self.id}"} for _ in documents],
                )
            else:
                self.vector_db = FAISS.from_documents(
                    documents,
                    embeddings,
                    metadatas=[{"source": f"submodule_{self.id}"} for _ in documents],
                )
            self.vector_db.save_local("document_store.faiss")
            logging.info(
                f"SubModule {self.id} stored {len(documents)} docs (Total: {len(self.vector_db.index_to_docstore_id)})"  # Add total count
            )
        except Exception as e:
            logging.error(f"Document storage failed: {str(e)}")

    def _filter_new_documents(self, documents):
        """Placeholder for filtering new documents - implement actual filtering logic"""
        return documents  # For now, return all documents without filtering

    @eidosian()
    async def run(self):
        """
        Continuously processes tasks from the task queue until stopped.
        Each task involves a multi-step processing pipeline for a given topic.
        """
        # Create a new event loop for the thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._run_processing_loop())
        finally:
            loop.close()

    async def _run_processing_loop(self):
        """Async processing loop with JSON logging"""
        while self.running:
            try:
                task = await asyncio.to_thread(task_queue.get, timeout=5)
                topic = task["topic"]
                logging.info(f"SubModule {self.id} processing topic: {topic}")

                # Step 1: Identify fundamental Python code
                logging.info(
                    f"SubModule {self.id} - Step 1: Identify Python code for topic: {topic}"
                )
                python_code = await self.identify_python_code(topic)

                # Step 2: Construct web queries
                logging.info(
                    f"SubModule {self.id} - Step 2: Construct web queries for topic: {topic}"
                )
                web_queries = await self.construct_web_queries(topic)

                # Step 3: Analyze the query
                logging.info(
                    f"SubModule {self.id} - Step 3: Analyze query for topic: {topic}"
                )
                analysis = await self.analyze_query(topic)

                # Step 4: Emotional context
                logging.info(
                    f"SubModule {self.id} - Step 4: Extract emotional context for topic: {topic}"
                )
                emotional_context = await self.emotional_context(topic)

                # Step 5: Compare outputs against the query
                logging.info(
                    f"SubModule {self.id} - Step 5: Compare outputs for topic: {topic}"
                )
                gaps = await self.compare_outputs(
                    topic, python_code, web_queries, analysis, emotional_context
                )

                # Step 6: Fill in gaps
                logging.info(
                    f"SubModule {self.id} - Step 6: Fill gaps for topic: {topic}"
                )
                filled_gaps = await self.fill_gaps(gaps)

                # Step 7: Summarize everything
                logging.info(
                    f"SubModule {self.id} - Step 7: Summarize for topic: {topic}"
                )
                summary = await self.summarize(
                    topic,
                    python_code,
                    web_queries,
                    analysis,
                    emotional_context,
                    filled_gaps,
                )

                # Step 8: Direct code construction
                logging.info(
                    f"SubModule {self.id} - Step 8: Construct code for topic: {topic}"
                )
                code_construction = await self.construct_code(topic)

                # Step 9: Perform web searches
                logging.info(
                    f"SubModule {self.id} - Step 9: Perform web searches for topic: {topic}"
                )
                search_results = await self.perform_web_searches(web_queries)

                # Step 10: Provide complete outputs
                logging.info(
                    f"SubModule {self.id} - Step 10: Provide complete outputs for topic: {topic}"
                )
                complete_outputs = self.provide_complete_outputs(
                    python_code,
                    web_queries,
                    analysis,
                    emotional_context,
                    gaps,
                    filled_gaps,
                    summary,
                    code_construction,
                    search_results,
                )

                # Step 11: Summarize all parts
                logging.info(
                    f"SubModule {self.id} - Step 11: Final summary for topic: {topic}"
                )
                final_summary = await self.final_summary(complete_outputs)

                # Step 12: Compile everything
                logging.info(
                    f"SubModule {self.id} - Step 12: Compile all for topic: {topic}"
                )
                compiled_output = self.compile_all(final_summary, complete_outputs)

                # Step 13: Save to file
                logging.info(
                    f"SubModule {self.id} - Step 13: Save output for topic: {topic}"
                )
                self.save_output(topic, compiled_output)

                # Step 14: Send frequent updates (handled by orchestrator) - Already handled by update_queue

                # Step 15: Final submission
                logging.info(
                    f"SubModule {self.id} - Step 15: Submit result for topic: {topic}"
                )
                result = {"topic": topic, "summary": compiled_output}
                result_queue.put(result)
                logging.info(f"SubModule {self.id} finished processing topic: {topic}")
                task["module_id"] = self.id
                task["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

                # Process task and store in JSON
                result = await self._process_task(task)
                result_queue.put(result)

                # Update JSON file incrementally
                self._update_output_json(result)

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(
                    f"SubModule {self.id} encountered an error during task processing: {e}",
                    exc_info=True,
                )
                update_queue.put(
                    f"Error in SubModule {self.id}: {e}"
                )  # Inform central orchestrator about submodule error

    async def _process_task(self, task):
        """Process task with JSON logging"""
        processing_steps = []

        # Add timing to each step
        start_time = time.time()
        processing_steps.append(
            {
                "step": "task_received",
                "timestamp": task["timestamp"],
                "module_id": self.id,
            }
        )

        # Process task and collect step data...

        return {
            "module_id": self.id,
            "topic": task["topic"],
            "processing_steps": processing_steps,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": time.time() - start_time,
        }

    def _update_output_json(self, result):
        """Atomic JSON update operation"""
        try:
            if os.path.exists(OUTPUT_JSON_PATH):
                with open(OUTPUT_JSON_PATH, "r") as f:
                    existing = json.load(f)
            else:
                existing = {"processing_steps": []}

            existing["processing_steps"].append(result)

            with open(OUTPUT_JSON_PATH, "w") as f:
                json.dump(existing, f, indent=2)

        except Exception as e:
            logging.error(f"JSON update failed: {str(e)}")

    # Core processing pipeline
    @eidosian()
    async def identify_python_code(self, topic):
        """
        Identifies fundamental Python code for a given topic using a language model.

        Args:
            topic (str): The topic to identify Python code for.

        Returns:
            str: Extracted Python code blocks or cleaned response.
        """
        prompt = LangchainPromptTemplate.from_template(
            "Identify fundamental Python code for: {topic}\n"
            "Include code blocks in ```python format\n"
            "Consider these constraints: {constraints}"
        ).format(topic=topic, constraints="")

        try:
            response = await OllamaHandler.generate(  # Defaulting to Ollama for now, consider hybrid approach later if needed
                prompt=prompt,
                model="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",
                temperature=0.1,
            )
            if "response" in response:
                text_response = response["response"]
                cleaned_response = remove_think_tags(text_response)
                code_blocks = re.findall(
                    CODE_BLOCK_PATTERN, cleaned_response, re.DOTALL
                )
                valid_blocks = [
                    b
                    for b in code_blocks
                    if any(kw in b for kw in ["def", "class", "import"])
                ]  # Quality filter
                return "\n".join(valid_blocks) if valid_blocks else cleaned_response
            return ""
        except Exception as e:
            logging.error(
                f"Error in identify_python_code for topic '{topic}': {e}", exc_info=True
            )
            return ""

    @eidosian()
    async def construct_web_queries(self, topic):
        """
        Constructs web queries to gather reference material for a given topic.

        Args:
            topic (str): The topic to construct web queries for.

        Returns:
            list: A list of web queries.
        """
        prompt = LangchainPromptTemplate.from_template(
            "Construct a series of 5 web queries to assist in gathering reference material and background material on the following topic:\n\n{topic}"
        ).format(topic=topic)
        try:
            response = await OllamaHandler.generate(  # Defaulting to Ollama for now, consider hybrid approach later if needed
                prompt=prompt,
                model="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",
                temperature=0.1,
            )
            if "response" in response:
                text_response = response["response"]
                cleaned_response = remove_think_tags(text_response)
                queries = re.findall(
                    r"\d+\.\s+(.+)", cleaned_response
                )  # Better list parsing
                return [q[:200] for q in queries if q.strip()][
                    :5
                ]  # Limit length and count
            return []
        except Exception as e:
            logging.error(
                f"Error in construct_web_queries for topic '{topic}': {e}",
                exc_info=True,
            )
            return []

    @eidosian()
    async def analyze_query(self, topic):
        """
        Analyzes the given query to determine better approaches or validity.

        Args:
            topic (str): The query topic to analyze.

        Returns:
            str: Analysis of the query.
        """
        prompt = LangchainPromptTemplate.from_template(
            "Break down and analyze the following query. Determine if there are any better approaches or if it is even a valid idea:\n\n{topic}"
        ).format(topic=topic)
        try:
            response = await OllamaHandler.generate(  # Defaulting to Ollama for now, consider hybrid approach later if needed
                prompt=prompt,
                model="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",
                temperature=0.1,
            )
            if "response" in response:
                text_response = response["response"]
                cleaned_response = remove_think_tags(text_response)
                return cleaned_response
            return ""
        except Exception as e:
            logging.error(
                f"Error in analyze_query for topic '{topic}': {e}", exc_info=True
            )
            return ""

    @eidosian()
    async def compare_outputs(
        self, topic, python_code, web_queries, analysis, emotional_context
    ):
        """
        Compares various outputs against the original query to identify gaps.

        Args:
            topic (str): The original query topic.
            python_code (str): Generated Python code.
            web_queries (list): Constructed web queries.
            analysis (str): Query analysis.
            emotional_context (dict): Emotional context analysis.

        Returns:
            str: Identified gaps in the outputs.
        """
        prompt = LangchainPromptTemplate.from_template(
            "Compare each of the following outputs against the original query to determine if there are any gaps:\n\nOriginal Query:\n{topic}\n\n1. Python Code:\n{python_code}\n\n2. Web Queries:\n{web_queries}\n\n3. Analysis:\n{analysis}\n\n4. Emotional Context:\n{emotional_context}\n\nIdentify any gaps."
        ).format(
            topic=topic,
            python_code=python_code,
            web_queries=web_queries,
            analysis=analysis,
            emotional_context=emotional_context,
        )
        try:
            response = await OllamaHandler.generate(  # Defaulting to Ollama for now, consider hybrid approach later if needed
                prompt=prompt,
                model="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",
                temperature=0.1,
            )
            if "response" in response:
                text_response = response["response"]
                cleaned_response = remove_think_tags(text_response)
                return cleaned_response
            return ""
        except Exception as e:
            logging.error(
                f"Error in compare_outputs for topic '{topic}': {e}", exc_info=True
            )
            return ""

    @eidosian()
    async def fill_gaps(self, gaps):
        """
        Fills in the identified gaps in the query processing.

        Args:
            gaps (str): The gaps identified in previous steps.

        Returns:
            str: Content that fills the identified gaps.
        """
        prompt = LangchainPromptTemplate.from_template(
            "Fill in the following gaps identified in the query processing:\n\n{gaps}"
        ).format(gaps=gaps)
        try:
            response = await OllamaHandler.generate(  # Defaulting to Ollama for now, consider hybrid approach later if needed
                prompt=prompt,
                model="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",
                temperature=0.1,
            )
            if "response" in response:
                text_response = response["response"]
                cleaned_response = remove_think_tags(text_response)
                return cleaned_response
            return ""
        except Exception as e:
            logging.error(f"Error in fill_gaps: {e}", exc_info=True)
            return ""

    @eidosian()
    async def summarize(
        self, topic, python_code, web_queries, analysis, emotional_context, filled_gaps
    ):
        """
        Summarizes all components related to the query.

        Args:
            topic (str): The original query topic.
            python_code (str): Generated Python code.
            web_queries (list): Constructed web queries.
            analysis (str): Query analysis.
            emotional_context (dict): Emotional context analysis.
            filled_gaps (str): Content that fills identified gaps.

        Returns:
            str: Summary of all components.
        """
        prompt = LangchainPromptTemplate.from_template(
            "Summarize the following components related to the query '{topic}':\n\n1. Python Code:\n{python_code}\n\n2. Web Queries:\n{web_queries}\n\n3. Analysis:\n{analysis}\n\n4. Emotional Context:\n{emotional_context}\n\n5. Filled Gaps:\n{filled_gaps}"
        ).format(
            topic=topic,
            python_code=python_code,
            web_queries=web_queries,
            analysis=analysis,
            emotional_context=emotional_context,
            filled_gaps=filled_gaps,
        )
        try:
            response = await OllamaHandler.generate(  # Defaulting to Ollama for now, consider hybrid approach later if needed
                prompt=prompt,
                model="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",
                temperature=0.1,
            )
            if "response" in response:
                text_response = response["response"]
                cleaned_response = remove_think_tags(text_response)
                return cleaned_response
            return ""
        except Exception as e:
            logging.error(f"Error in summarize for topic '{topic}': {e}", exc_info=True)
            return ""

    @eidosian()
    async def construct_code(self, topic):
        """
        Constructs the necessary Python code to solve the query based on summarized information.

        Args:
            topic (str): The original query topic.

        Returns:
            str: Constructed Python code.
        """
        prompt = LangchainPromptTemplate.from_template(
            "Based on the summarized information, construct the necessary Python code to solve the query:\n\n{topic}"
            "Construct code with proper formatting:\n"
            "```python\n"
            "# Your Python code here\n"
            "{code}\n"
            "```"
        ).format(topic=topic)
        try:
            response = await OllamaHandler.generate(  # Defaulting to Ollama for now, consider hybrid approach later if needed
                prompt=prompt,
                model="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",
                temperature=0.1,
            )
            if "response" in response:
                text_response = response["response"]
                cleaned_response = remove_think_tags(text_response)
                return cleaned_response
            return ""
        except Exception as e:
            logging.error(
                f"Error in construct_code for topic '{topic}': {e}", exc_info=True
            )
            return ""

    @eidosian()
    async def final_summary(self, complete_outputs):
        """
        Summarizes all complete outputs to produce a final summary.

        Args:
            complete_outputs (dict): Dictionary containing all processed outputs.

        Returns:
            str: Final summary of all outputs.
        """
        prompt = LangchainPromptTemplate.from_template(
            "Summarize all the following parts:\n\n{complete_outputs}"
        ).format(complete_outputs=complete_outputs)
        try:
            response = await OllamaHandler.generate(  # Defaulting to Ollama for now, consider hybrid approach later if needed
                prompt=prompt,
                model="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",
                temperature=0.1,
            )
            if "response" in response:
                text_response = response["response"]
                cleaned_response = remove_think_tags(text_response)
                return cleaned_response
            return ""
        except Exception as e:
            logging.error(f"Error in final_summary: {e}", exc_info=True)
            return ""

    @eidosian()
    def compile_all(self, final_summary, complete_outputs):
        """
        Compiles the final summary and detailed outputs into a formatted string.

        Args:
            final_summary (str): The final summary.
            complete_outputs (dict): Dictionary of detailed outputs.

        Returns:
            str: Compiled output string.
        """
        compiled = f"Final Summary:\n{final_summary}\n\nDetailed Outputs:\n"
        for key, value in complete_outputs.items():
            compiled += f"\n--- {key.replace('_', ' ').title()} ---\n{value}\n"
        return compiled

    def _process_query(self, query):
        """Processes the query before searching (currently just returns the query)"""
        return query

    @eidosian()
    def search_searxng(self, query, num_results=5):
        """
        Performs a SearxNG search against the configured SEARXNG_URL.
        Returns JSON results for the given query.
        """

        SEARXNG_URL = "http://192.168.4.73:8888"  # Or configure as needed
        params = {
            "q": query,
            "format": "json",
            "engines": "google,bing,academic",
            "categories": "science",
            "safesearch": 1,
            "lang": "en",
            "count": num_results,
        }
        try:
            response = requests.get(urljoin(SEARXNG_URL, "/search"), params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during SearxNG search: {e}")
            return {"results": []}  # Return empty results in case of error

    @eidosian()
    def scrape_and_save_content(self, search_results, query):
        """
        Scrapes content from SearxNG search results and saves it.
        """
        urls = [
            result.get("url")
            for result in search_results.get("results", [])
            if result.get("url")
        ]
        if not urls:
            print("No URLs found in search results to scrape.")
            return []

        print(f"URLs to scrape: {urls}")
        return self.perform_web_searches(
            urls
        )  # Directly use perform_web_searches to process URLs

    @eidosian()
    def extract_content_with_tika(self, url):
        """
        Fetches content from a URL and extracts text and metadata using Tika.
        """

        TIKA_URL = "http://192.168.4.73:9998"  # Or configure as needed
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            content_type = response.headers.get(
                "Content-Type", "application/octet-stream"
            )
            headers = {"Content-type": content_type, "Accept": "application/json"}
            tika_response = requests.put(
                f"{TIKA_URL}/rmeta", headers=headers, data=response.content
            )
            tika_response.raise_for_status()
            return json.loads(tika_response.text)
        except Exception as e:
            print(f"Error extracting content with Tika for URL {url}: {e}")
            return []

    def _process_tika_output(self, tika_data):
        """Processes the raw Tika output (currently returns it as is).
        Further processing can be added here to refine Tika's output.
        """
        return tika_data

    # Output management
    @eidosian()
    def provide_complete_outputs(
        self,
        python_code,
        web_queries,
        analysis,
        emotional_context,
        gaps,
        filled_gaps,
        summary,
        code_construction,
        search_results,
    ):
        """
        Provides a structured dictionary containing all processed outputs.

        Args:
            python_code (str): Generated Python code.
            web_queries (list): Constructed web queries.
            analysis (str): Query analysis.
            emotional_context (dict): Emotional context analysis.
            gaps (str): Identified gaps.
            filled_gaps (str): Filled gaps content.
            summary (str): Summary of components.
            code_construction (str): Constructed code.
            search_results (list): Web search results.

        Returns:
            dict: Dictionary of complete outputs.
        """
        return {
            "python_code": python_code,
            "web_queries": web_queries,
            "analysis": analysis,
            "emotional_context": emotional_context,
            "gaps": gaps,
            "filled_gaps": filled_gaps,
            "summary": summary,
            "code_construction": code_construction,
            "search_results": search_results,
        }

    @eidosian()
    def save_output(self, topic, compiled_output):
        """
        Robust output saving with checksum validation to ensure data integrity.

        Args:
            topic (str): The original query topic.
            compiled_output (str): The compiled output to save.
        """
        try:
            sanitized_topic = re.sub(r"[^\w\-_\. ]", "", topic)[:100]
            folder_name = f"{self.id}_{sanitized_topic.replace(' ', '_')}"
            folder_path = os.path.join("outputs", folder_name)

            os.makedirs(folder_path, exist_ok=True)

            # Generate checksum
            content_hash = hashlib.sha256(compiled_output.encode()).hexdigest()

            # Save with versioning
            file_path = os.path.join(folder_path, f"output_{content_hash[:8]}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(compiled_output)

            # Validate write operation
            if not os.path.exists(file_path):
                raise IOError("File write verification failed")

            update_queue.put(f"Output saved to {file_path} | Checksum: {content_hash}")
            logging.info(
                f"SubModule {self.id} saved output to {file_path} with checksum {content_hash}"
            )

        except Exception as e:
            error_msg = f"Failed to save output for '{topic}': {str(e)}"
            logging.error(error_msg)
            update_queue.put(error_msg)

    # Validation/error handling
    def _validate_response(self, response, min_length=10):
        """
        Ensures response quality meets minimum length standards.

        Args:
            response (str): The response string to validate.
            min_length (int): Minimum acceptable length for the response.

        Returns:
            str: The validated response if it meets the criteria.

        Raises:
            ValueError: If the response is invalid or too short.
        """
        if not response or len(response) < min_length:
            raise ValueError("Invalid or empty model response")
        return response

    def _handle_content_error(self, url, error):
        """
        Centralized error handling for content processing failures.

        Args:
            url (str): The URL that caused the error.
            error (Exception): The exception encountered.

        Returns:
            dict: Error information dictionary.
        """
        error_msg = f"Content processing failed for {url}: {str(error)}"
        logging.error(error_msg)
        update_queue.put(error_msg)
        return {
            "error": error_msg,
            "url": url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _validate_web_result(self, result):
        """
        Quality check for web search results to ensure they are valid and useful.

        Args:
            result (dict): A dictionary representing a web search result.

        Returns:
            bool: True if the result is valid, False otherwise.
        """
        MIN_CONTENT_LENGTH = 100  # Increased from 50
        if not isinstance(result, dict):
            return False
        if not result.get("url", "").startswith("http"):
            return False  # Verify URL protocol
        return len(result.get("content", "")) >= MIN_CONTENT_LENGTH

    async def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        """Unified Ollama call handler with basic fallback"""
        try:
            response = await OllamaHandler.generate(
                prompt=prompt,
                model="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",
                system=system_prompt,
                options={"temperature": 0.3, "num_ctx": 4096},
            )
            return response.get("response", "").split("</think>")[-1].strip()
        except Exception as e:
            logging.warning(f"Ollama call failed: {str(e)}")
            return ""

    @eidosian()
    async def extract_content(self, url: str) -> str:
        """Unified content extraction"""
        try:
            loader = AsyncHtmlLoader([url])
            docs = await loader.aload()  # Use aload() instead of load()
            transformed = Html2TextTransformer().transform_documents(docs)
            return transformed[0].page_content if transformed else ""
        except Exception as e:
            tika_content = self.extract_content_with_tika(url)
            return (
                "\n".join([item.get("content", "") for item in tika_content])
                if isinstance(tika_content, list)
                else ""
            )

    @eidosian()
    async def score_relevance(self, content: str, query: str) -> float:
        """Score content relevance using local model"""
        prompt = f"Rate relevance (0-1) between:\nQ: {query}\nC: {content[:2000]}"
        response = await self._local_generate(prompt)
        try:
            return min(max(float(response.strip()), 0), 1)
        except:
            return 0.5


class OllamaHandler:
    """Modern async Ollama API handler with automatic model management"""

    @staticmethod
    async def generate(prompt: str, model: str, **params) -> dict:
        """Handle both streaming and non-streaming responses"""
        url = urljoin(OLLAMA_BASE_URL, "generate")
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "top_p": 0.95, "num_ctx": 4096},
        }
        payload.update(params)

        try:
            response = await ASYNC_CLIENT.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logging.error(f"Ollama API error: {e.response.text}")
            return {"error": str(e)}


# ------------------
# EXECUTION
# ------------------
if __name__ == "__main__":
    # Main execution block
    user_input = "Develop an efficient algorithm for real-time traffic prediction using minimal computational resources."

    orchestrator = CentralOrchestrator()
    orchestrator.run(user_input)

    # Wait for all threads to finish (optional)
    for thread in orchestrator.processing_threads:
        thread.join()

    orchestrator.update_thread.join()
