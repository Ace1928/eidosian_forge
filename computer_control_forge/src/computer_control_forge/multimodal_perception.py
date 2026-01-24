#!/usr/bin/env python3
"""
Multi-Modal Perception System for Eidos

Combines multiple perception sources:
- Visual: Screenshot + OCR
- System: CPU, Memory, Disk I/O
- Window: Active window, window list via D-Bus
- Cursor: Position via KWin scripting (when available)

Outputs a unified world state for reasoning.
"""

import subprocess
import time
import json
import os
import psutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

# Constants
SCREENSHOT_PATH = "/tmp/eidos_perception.png"
OCR_PATH = "/tmp/eidos_ocr"
STATE_PATH = "/tmp/eidos_world_state.json"

@dataclass
class TextElement:
    """A detected text element with position."""
    text: str
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    confidence: float = 0.0

@dataclass 
class WindowInfo:
    """Information about a window."""
    id: str
    caption: str
    active: bool = False

@dataclass
class SystemState:
    """System metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_read_kb: float
    disk_write_kb: float

@dataclass
class VisualState:
    """Visual perception from screenshot + OCR."""
    screenshot_path: str
    timestamp: str
    elements: List[TextElement] = field(default_factory=list)
    raw_text: str = ""

@dataclass
class CursorState:
    """Cursor position."""
    x: int
    y: int
    method: str  # How we got this: "kwin", "inference", "unknown"

@dataclass
class WorldState:
    """Complete perception of the world."""
    timestamp: str
    system: SystemState
    visual: VisualState
    windows: List[WindowInfo] = field(default_factory=list)
    active_window: Optional[str] = None
    cursor: Optional[CursorState] = None
    perception_time_ms: float = 0.0


class MultiModalPerception:
    """Multi-modal perception system."""
    
    def __init__(self):
        self._last_disk = psutil.disk_io_counters()
        self._last_disk_time = time.time()
    
    def _get_system_state(self) -> SystemState:
        """Get system metrics."""
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        
        # Disk I/O delta
        current_disk = psutil.disk_io_counters()
        now = time.time()
        dt = now - self._last_disk_time
        if dt > 0:
            read_kb = (current_disk.read_bytes - self._last_disk.read_bytes) / 1024 / dt
            write_kb = (current_disk.write_bytes - self._last_disk.write_bytes) / 1024 / dt
        else:
            read_kb = write_kb = 0
        self._last_disk = current_disk
        self._last_disk_time = now
        
        return SystemState(
            cpu_percent=round(cpu, 1),
            memory_percent=round(mem.percent, 1),
            memory_available_gb=round(mem.available / (1024**3), 2),
            disk_read_kb=round(read_kb, 1),
            disk_write_kb=round(write_kb, 1)
        )
    
    def _get_screenshot(self) -> str:
        """Capture screenshot."""
        try:
            subprocess.run(
                ['spectacle', '-b', '-f', '-n', '-o', SCREENSHOT_PATH],
                capture_output=True, timeout=5
            )
            return SCREENSHOT_PATH
        except Exception as e:
            return ""
    
    def _get_ocr(self, image_path: str) -> tuple[List[TextElement], str]:
        """OCR the screenshot, return elements and raw text."""
        elements = []
        raw_text = ""
        
        if not image_path or not Path(image_path).exists():
            return elements, raw_text
        
        try:
            # Get raw text
            result = subprocess.run(
                ['tesseract', image_path, 'stdout'],
                capture_output=True, text=True, timeout=15
            )
            raw_text = result.stdout
            
            # Get word boxes
            subprocess.run(
                ['tesseract', image_path, OCR_PATH, '-c', 'tessedit_create_tsv=1'],
                capture_output=True, timeout=15
            )
            
            tsv_path = f"{OCR_PATH}.tsv"
            if Path(tsv_path).exists():
                with open(tsv_path, 'r') as f:
                    lines = f.readlines()[1:]  # Skip header
                    for line in lines:
                        parts = line.strip().split('\t')
                        if len(parts) >= 12 and parts[11].strip():
                            try:
                                x = int(parts[6])
                                y = int(parts[7])
                                w = int(parts[8])
                                h = int(parts[9])
                                conf = float(parts[10]) if parts[10] != '-1' else 0
                                text = parts[11]
                                
                                elements.append(TextElement(
                                    text=text,
                                    x=x, y=y, width=w, height=h,
                                    center_x=x + w//2,
                                    center_y=y + h//2,
                                    confidence=conf
                                ))
                            except (ValueError, IndexError):
                                pass
        except Exception as e:
            pass
        
        return elements, raw_text
    
    def _get_windows(self) -> tuple[List[WindowInfo], Optional[str]]:
        """Get window list and active window via KWin D-Bus."""
        windows = []
        active = None
        
        try:
            # Get active window
            result = subprocess.run(
                ['qdbus', 'org.kde.KWin', '/KWin', 'org.kde.KWin.activeWindow'],
                capture_output=True, text=True, timeout=2
            )
            active_id = result.stdout.strip()
            
            if active_id:
                # Get caption
                result = subprocess.run(
                    ['qdbus', 'org.kde.KWin', active_id, 'org.kde.KWin.caption'],
                    capture_output=True, text=True, timeout=2
                )
                active = result.stdout.strip()
                windows.append(WindowInfo(id=active_id, caption=active, active=True))
        except Exception as e:
            pass
        
        return windows, active
    
    def _get_cursor(self) -> Optional[CursorState]:
        """Get cursor position via KWin scripting."""
        try:
            script = 'console.log("EIDOS_CURSOR:" + workspace.cursorPos.x + "," + workspace.cursorPos.y);'
            script_path = '/tmp/eidos_cursor.js'
            with open(script_path, 'w') as f:
                f.write(script)
            
            result = subprocess.run(
                ['qdbus', 'org.kde.KWin', '/Scripting',
                 'org.kde.kwin.Scripting.loadScript', script_path],
                capture_output=True, text=True, timeout=2
            )
            script_id = result.stdout.strip()
            
            if script_id and script_id != '-1':
                subprocess.run(
                    ['qdbus', 'org.kde.KWin', f'/Scripting/Script{script_id}',
                     'org.kde.kwin.Script.run'],
                    capture_output=True, timeout=2
                )
                time.sleep(0.15)
                
                result = subprocess.run(
                    ['journalctl', '--user', '-u', 'plasma-kwin_wayland',
                     '-n', '20', '--no-pager', '-o', 'cat'],
                    capture_output=True, text=True, timeout=2
                )
                
                for line in reversed(result.stdout.split('\n')):
                    if 'EIDOS_CURSOR:' in line:
                        coords = line.split('EIDOS_CURSOR:')[1].split()[0]
                        x, y = map(int, coords.split(','))
                        
                        # Cleanup
                        subprocess.run(
                            ['qdbus', 'org.kde.KWin', f'/Scripting/Script{script_id}',
                             'org.kde.kwin.Script.stop'],
                            capture_output=True, timeout=2
                        )
                        return CursorState(x=x, y=y, method="kwin")
        except Exception as e:
            pass
        
        return None
    
    def perceive(self, include_visual: bool = True) -> WorldState:
        """
        Perform full multi-modal perception.
        
        Args:
            include_visual: Whether to capture screenshot and OCR (slower but comprehensive)
        
        Returns:
            Complete world state
        """
        start = time.time()
        timestamp = datetime.now().isoformat()
        
        # Fast sensors first
        system = self._get_system_state()
        windows, active_window = self._get_windows()
        cursor = self._get_cursor()
        
        # Visual perception (slower)
        if include_visual:
            screenshot = self._get_screenshot()
            elements, raw_text = self._get_ocr(screenshot)
            visual = VisualState(
                screenshot_path=screenshot,
                timestamp=timestamp,
                elements=elements,
                raw_text=raw_text
            )
        else:
            visual = VisualState(screenshot_path="", timestamp=timestamp)
        
        perception_time = (time.time() - start) * 1000
        
        state = WorldState(
            timestamp=timestamp,
            system=system,
            visual=visual,
            windows=windows,
            active_window=active_window,
            cursor=cursor,
            perception_time_ms=round(perception_time, 1)
        )
        
        # Save state for external access
        self._save_state(state)
        
        return state
    
    def _save_state(self, state: WorldState):
        """Save world state to JSON for external access."""
        def to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [to_dict(i) for i in obj]
            else:
                return obj
        
        with open(STATE_PATH, 'w') as f:
            json.dump(to_dict(state), f, indent=2)
    
    def find_element(self, *search_terms, min_confidence: float = 50.0) -> Optional[TextElement]:
        """Find a text element containing any of the search terms."""
        state = self.perceive(include_visual=True)
        
        for element in state.visual.elements:
            if element.confidence < min_confidence:
                continue
            text_lower = element.text.lower()
            for term in search_terms:
                if term.lower() in text_lower:
                    return element
        return None
    
    def find_all(self, *search_terms, min_confidence: float = 50.0) -> List[TextElement]:
        """Find all text elements matching search terms."""
        state = self.perceive(include_visual=True)
        matches = []
        
        for element in state.visual.elements:
            if element.confidence < min_confidence:
                continue
            text_lower = element.text.lower()
            for term in search_terms:
                if term.lower() in text_lower:
                    matches.append(element)
                    break
        
        return matches


if __name__ == "__main__":
    print("Testing Multi-Modal Perception System...")
    perception = MultiModalPerception()
    
    # Full perception
    state = perception.perceive(include_visual=True)
    
    print(f"\n=== World State @ {state.timestamp} ===")
    print(f"Perception took: {state.perception_time_ms}ms")
    print(f"\nSystem:")
    print(f"  CPU: {state.system.cpu_percent}%")
    print(f"  Memory: {state.system.memory_percent}%")
    print(f"  Disk R/W: {state.system.disk_read_kb}/{state.system.disk_write_kb} KB/s")
    print(f"\nActive Window: {state.active_window}")
    print(f"Cursor: {state.cursor}")
    print(f"\nVisual Elements: {len(state.visual.elements)} detected")
    
    # Find specific elements
    chatgpt_elements = perception.find_all('chatgpt', 'ask', 'anything')
    print(f"\nChatGPT-related elements: {len(chatgpt_elements)}")
    for el in chatgpt_elements[:5]:
        print(f"  '{el.text}' at ({el.center_x}, {el.center_y})")
