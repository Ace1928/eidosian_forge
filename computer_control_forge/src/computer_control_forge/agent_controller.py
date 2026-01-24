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
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{ts}] [{level}] {message}"
        print(entry)
        with open(self.log_path, 'a') as f:
            f.write(entry + "\n")
    
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
        
        # Filter by region if specified
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
            # Return highest confidence match
            matches.sort(key=lambda x: x.confidence, reverse=True)
            match = matches[0]
            self.log(f"Found '{match.text}' at ({match.center_x}, {match.center_y})")
            return match
        
        return None
    
    def click_on(self, *terms, region: str = None, verify: bool = True) -> bool:
        """
        Find an element and click on it.
        
        Args:
            terms: Text to search for
            region: Optional region filter
            verify: Whether to verify the click worked
        """
        element = self.find(*terms, region=region)
        if not element:
            self.log(f"Cannot click - element not found: {terms}", "ERROR")
            return False
        
        self.log(f"Clicking on '{element.text}' at ({element.center_x}, {element.center_y})")
        success = self.actions.click_at(element.center_x, element.center_y)
        
        if verify and success:
            time.sleep(0.3)
            # Re-perceive to verify state changed
            self.perceive()
        
        return success
    
    def type_in(self, text: str, target: str = None, region: str = None) -> bool:
        """
        Type text, optionally clicking a target first.
        
        Args:
            text: Text to type
            target: Optional element to click first
            region: Region filter for target
        """
        if target:
            if not self.click_on(target, region=region):
                return False
            time.sleep(0.2)
        
        self.log(f"Typing: {text[:50]}..." if len(text) > 50 else f"Typing: {text}")
        return self.actions.type_text(text)
    
    def send_message(self, text: str) -> bool:
        """Type text and press enter."""
        if self.type_in(text):
            time.sleep(0.1)
            return self.actions.press_key('enter')
        return False
    
    def wait_for(self, *terms, timeout: float = 10.0, region: str = None) -> Optional[TextElement]:
        """
        Wait for an element to appear.
        
        Args:
            terms: Text to wait for
            timeout: Maximum time to wait
            region: Region filter
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
    
    def analyze_screen(self) -> Dict[str, Any]:
        """
        Analyze the current screen and return a summary.
        """
        if not self.state:
            self.perceive()
        
        # Categorize elements by region
        regions = {
            'top': [],
            'middle': [],
            'bottom': [],
            'left_sidebar': [],
            'right_sidebar': [],
        }
        
        for el in self.state.visual.elements:
            if el.confidence < 60:
                continue
            
            y, x = el.center_y, el.center_x
            
            if y < 200:
                regions['top'].append(el.text)
            elif y > 800:
                regions['bottom'].append(el.text)
            else:
                regions['middle'].append(el.text)
            
            if x < 300:
                regions['left_sidebar'].append(el.text)
            elif x > 1600:
                regions['right_sidebar'].append(el.text)
        
        return {
            'active_window': self.state.active_window,
            'total_elements': len(self.state.visual.elements),
            'regions': {k: len(v) for k, v in regions.items()},
            'system': {
                'cpu': self.state.system.cpu_percent,
                'memory': self.state.system.memory_percent,
            }
        }
    
    def describe_view(self) -> str:
        """Generate a human-readable description of what's on screen."""
        analysis = self.analyze_screen()
        
        desc = f"Active window: {analysis['active_window'] or 'Unknown'}\n"
        desc += f"Detected {analysis['total_elements']} text elements\n"
        desc += f"System: CPU {analysis['system']['cpu']}%, Memory {analysis['system']['memory']}%\n"
        desc += f"Regions: {analysis['regions']}"
        
        return desc


if __name__ == "__main__":
    print("=== Eidos Agent Controller Test ===\n")
    
    agent = AgentController()
    
    # Perceive
    agent.perceive()
    
    # Analyze
    print("\n" + agent.describe_view())
    
    # Find ChatGPT elements
    print("\nLooking for ChatGPT UI elements...")
    
    # Find input area (should be at bottom)
    input_el = agent.find('ask', 'anything', 'message', region='bottom')
    if input_el:
        print(f"Found input at ({input_el.center_x}, {input_el.center_y})")
    
    # Find attachment/paperclip
    attach_el = agent.find('attach', '+', 'file')
    if attach_el:
        print(f"Found attach at ({attach_el.center_x}, {attach_el.center_y})")
