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

```
# File Summary

The `agent_controller.py` file is the main control loop for Eidos, responsible for perceiving the world (fast, multi-modal), reasoning about what to do (slow, deliberate), acting (precise, verified), verifying results (perception again), and iterating.

---

# API Documentation

## Functions
- **perceive**: Perceives the world by fetching visual perception data.
  - Parameters: `full` (boolean) — Include full visual perception (slower but comprehensive).
  
- **find**: Finds UI elements based on text input.
  - Parameters:
    - *terms*: Text to search for.
    - *region*: Optional region filter ('top', 'bottom', 'left', 'right').
  - Returns: The first matching `TextElement` or None if none found.

- **click_on**: Clicks on an element based on text input, with verification of click success.
  - Parameters:
    - *terms*: Text to search for.
    - *region*: Optional region filter (default is 'bottom').
    - *verify*: Whether to verify the click worked (default is True).
  
- **type_in**: Types text and clicks a target element if specified, with verification of typing success.
  - Parameters:
    - `text`: Text to type.
    - `target`: Optional element to click first.
    - `region`: Region filter for target (optional).

- **send_message**: Sends typed message via the agent, with enter key press.
  
- **wait_for**: Waits for an element to appear on the screen within a timeout period.
  - Parameters:
    - *terms*: Text to wait for.
    - *timeout*: Maximum time to wait in seconds (default is 10.0).
  
- **analyze_screen**: Analyzes the current screen and returns a summary of detected elements, regions, system metrics, and active window.
- **describe_view**: Generates a human-readable description of what's on screen based on analysis.

---

# Current Status

The codebase is well-defined with clear functions for perceiving the world, finding UI elements, interacting with them, and describing their appearance. The state management (`state`) and log handling are robust. 

### Future Directions
- **Error Handling**: Enhance error logging to handle edge cases more gracefully.
- **Logging Frequency**: Optimize logging frequency based on user requirements (e.g., real-time vs. batch).
- **Performance Tuning**: Improve performance by optimizing the `perceive` and `analyze_screen` functions.

---

```python
#!/usr/bin/env python3

"""
Eidos Agent Controller

The main control loop that:
1. Perceives the world (fast, multi-modal)
2. Reasons about what to do (slow, deliberate)
3. Acts (precise, verified)
4. Verifies results (perception again)
5. Iterates

This is where agency happens.
"""

import sys
import time
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass

sys.path.insert(0, '/home/lloyd/eidosian_forge/computer_control_forge/src/computer_control_forge')

from multimodal_perception import MultiModalPerception, WorldState, TextElement
from actions import Actions


@dataclass
class Task:
    """A task to be executed."""
    name: str
    description: str
    steps: List[Dict[str, Any]]
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[str] = None

# Class for the Eidos Agent Controller.
class AgentController:
    """
    The Eidos Agent Controller.

    Observes, reasons, acts, verifies.
    """
    
    def __init__(self):
        self.perception = MultiModalPerception()
        self.actions = Actions()
        self.state: Optional[WorldState] = None
        self.history: List[Dict] = []
        self.log_path = "/tmp/eidos_agent.log"
    
    # Logging function.
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{ts}] [{level}] {message}"
        print(entry)
        with open(self.log_path, 'a') as f:
            f.write(entry + "\n")
    
    # Perceive world function.
    def perceive(self, full: bool = True) -> WorldState:
        """
        Perceive the world.

        Args:
            full: Include visual perception (slower but comprehensive)
        """
        self.log("Perceiving world...")
        self.state = self.perception.perceive(include_visual=full)
        self.log(f"Perception complete in {self.state.perception_time_ms}ms - "
                f"{len(self.state.visual.elements)} elements detected")
        return self.state
    
    # Find UI element function.
    def find(self, *terms, region: str = None) -> Optional[TextElement]:
        """
        Find a UI element by text.

        Args:
            terms: Text to search for
            region: Optional region filter ('top', 'bottom', 'left', 'right')
        """
        if not self.state:
            self.perceive()
        
        matches = []
        for el in self.state.visual.elements:
            if el.confidence < 50:
                continue
            text_lower = el.text.lower()
            for term in terms:
                if term.lower() in text_lower:
                    matches.append(el)
                    break
        
        if not matches:
            self.log(f"Could not find element matching: {terms}")
            return None
        
        # Filter by region if specified.
        if region:
            if region == 'bottom':
                matches = [m for m in matches if m.center_y > 700]
            elif region == 'top':
                matches = [m for m in matches if m.center_y < 300]
            elif region == 'left':
                matches = [m for m in matches if m.center_x < 500]
            elif region == 'right':
                matches = [m for m in matches if m.center_x > 1400]
        
        if matches:
            # Return highest confidence match.
            matches.sort(key=lambda x: x.confidence, reverse=True)
            match = matches[0]
            self.log(f"Found '{match.text}' at ({match.center_x}, {match.center_y})")
            return match
        
        return None
    
    # Click on UI element function.
    def click_on(self, *terms, region: str = None, verify: bool = True) -> bool:
        """
        Find an element and click it.

        Args:
            terms: Text to search for
            region: Optional region filter
            verify: Whether to verify the click worked (default is True)
        """
        element = self.find(*terms, region=region)
        if not element:
            self.log(f"Cannot click - element not found: {terms}", "ERROR")
            return False
        
        self.log(f"Clicking on '{element.text}' at ({element.center_x}, {element.center_y})")
        success = self.actions.click_at(element.center_x, element.center_y)
        
        if verify and success:
            time.sleep(0.3)
            # Re-perceive to verify state changed.
            self.perceive()
        
        return success
    
    # Type in text function.
    def type_in(self, text: str, target: str = None, region: str = None) -> bool:
        """
        Type text and optionally click a target first.

        Args:
            text: Text to type
            target: Optional element to click first
            region: Region filter for the target (optional)
        """
        if target:
            if not self.click_on(target, region=region):
                return False
            time.sleep(0.2)
        
        self.log(f"Typing: {text[:50]}..." if len(text) > 50 else f"Typing: {text}")
        return self.actions.type_text(text)
    
    # Send message function.
    def send_message(self, text: str) -> bool:
        """Type text and press enter."""
        if self.type_in(text):
            time.sleep(0.1)
            return self.actions.press_key('enter')
        return False
    
    # Wait for UI element to appear function.
    def wait_for(self, *terms, timeout: float = 10.0, region: str = None) -> Optional[TextElement]:
        """
        Wait for an element to appear.

        Args:
            terms: Text to wait for
            timeout: Maximum time to wait (default is 10 seconds)
            region: Region filter for the target (optional)
        """
        start = time.time()
        while time.time() - start < timeout:
            self.perceive()
            element = self.find(*terms, region=region)
            if element:
                return element
            time.sleep(0.5)
        
        self.log(f"Timeout waiting for: {terms}", "WARN")
        return None
    
    # Analyze screen function.
    def analyze_screen(self) -> Dict[str, Any]:
        """
        Analyze the current screen and return a summary.

        Returns:
            A dictionary with system metrics, region stats, and active window information.
        """
        if not self.state:
            self.perceive()

[ Prompt: 46.2 t/s | Generation: 5.3 t/s ]

Exiting...