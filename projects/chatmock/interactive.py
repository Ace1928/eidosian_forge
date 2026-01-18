from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from .config import BASE_INSTRUCTIONS
from .upstream import start_upstream_request, normalize_model_name
from .utils import convert_chat_messages_to_responses_input
from .logging_config import get_logger

logger = get_logger(__name__)

class InteractiveChat:
    def __init__(self, model: str = "gpt-5", verbose: bool = False):
        self.model = normalize_model_name(model)
        self.verbose = verbose
        self.history = InMemoryHistory()
        self.session = PromptSession(history=self.history)
        self.messages: List[Dict[str, str]] = []
        self.style = Style.from_dict({
            'user': '#00ff00 bold',
            'assistant': '#00aaff',
            'system': '#ff00ff italic',
            'error': '#ff0000 bold',
        })
        self.stop_event = threading.Event()

    def _print_stream(self, upstream_response) -> str:
        """Reads from upstream response stream and prints to stdout. Returns full text."""
        full_text = ""
        try:
            for raw_line in upstream_response.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, (bytes, bytearray)) else raw_line
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if not data or data == "[DONE]":
                    if data == "[DONE]":
                        break
                    continue
                
                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue

                kind = evt.get("type")
                delta = ""
                
                if kind == "response.output_text.delta":
                    delta = evt.get("delta") or ""
                elif kind == "response.reasoning_text.delta":
                    # Optionally handle reasoning text differently
                    delta = evt.get("delta") or ""
                
                if delta:
                    print(delta, end="", flush=True)
                    full_text += delta
                    
        except Exception as e:
            logger.error(f"Error reading stream: {e}")
            print(f"\n[Error reading stream: {e}]", file=sys.stderr)
        
        print() # Newline at end
        return full_text

    def run(self) -> int:
        print(f"Starting interactive chat with model: {self.model}")
        print("Type 'exit' or 'quit' to end session.")
        print("-" * 50)

        while True:
            try:
                user_input = self.session.prompt(HTML('<user>You:</user> '), style=self.style)
                user_input = user_input.strip()

                if not user_input:
                    continue
                
                if user_input.lower() in ('exit', 'quit'):
                    print("Goodbye!")
                    return 0

                self.messages.append({"role": "user", "content": user_input})
                
                # logic fix: upstream.py now handles missing flask request context gracefully. 
                
                # Prepare input for upstream
                input_items = convert_chat_messages_to_responses_input(self.messages)

                print("Assistant: ", end="", flush=True)
                upstream, error_resp = start_upstream_request(
                    model=self.model,
                    input_items=input_items,
                    instructions=BASE_INSTRUCTIONS, # Use default instructions
                    tools=[],
                    tool_choice="auto",
                )

                if error_resp:
                    # error_resp is a Flask Response object
                    try:
                        err_data = json.loads(error_resp.get_data(as_text=True))
                        print(f"\nError: {err_data}", style=self.style)
                    except:
                        print(f"\nError: {error_resp.status_code}", style=self.style)
                elif upstream is not None:
                    if upstream.status_code >= 400:
                         print(f"\nUpstream Error: {upstream.status_code} - {upstream.text}")
                    else:
                        response_text = self._print_stream(upstream)
                        self.messages.append({"role": "assistant", "content": response_text})
                else:
                    print("\nNo response from upstream.")

            except KeyboardInterrupt:
                print("\nInterrupted. Type 'exit' to quit.")
                continue
            except EOFError:
                return 0
            except Exception as e:
                logger.exception("An error occurred during chat session")
                print(f"An error occurred: {e}")