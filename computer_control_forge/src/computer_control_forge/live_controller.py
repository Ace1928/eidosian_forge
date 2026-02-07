#!/usr/bin/env python3
"""
Integrated Live Controller

Full perception + mouse + keyboard control with multi-modal feedback.
Designed for AI agents to have real-time awareness of their actions.

Author: Eidos
Version: 1.0.0
"""

import sys
import os
import time
import json
import subprocess
import math
import random
import threading
import uuid
from typing import Tuple, Optional, List, Dict, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("YDOTOOL_SOCKET", "/tmp/.ydotool_socket")

from cursor_position import KWinCursorPosition
from feedback_system import FeedbackHub, MCPChannel
from perception.window_manager import WindowManager


@dataclass
class ControllerConfig:
    """Configuration for the live controller."""
    # Feedback channels
    enable_stdout: bool = True
    enable_file: bool = True
    file_path: str = "/tmp/eidos_live.jsonl"
    enable_fifo: bool = False
    fifo_path: str = "/tmp/eidos_feedback"
    enable_socket: bool = False
    socket_path: str = "/tmp/eidos_feedback.sock"
    enable_agent_bus: bool = False
    agent_bus_state_dir: str = "state"
    enable_pipeline_trace: bool = False
    
    # Mouse settings
    mouse_tolerance: float = 8.0
    mouse_max_iterations: int = 30
    mouse_human_like: bool = True
    
    # Update rates
    position_update_hz: float = 20.0
    window_update_hz: float = 2.0
    
    # Screen
    screen_width: int = 1920
    screen_height: int = 1080
    edge_margin: int = 5


class LiveController:
    """
    Integrated controller with live feedback on all operations.
    
    Provides real-time visibility into:
    - Mouse movements (position, velocity, edge detection)
    - Keyboard input (keys pressed, text typed)
    - Window state (active window, focus changes)
    - Screen perception (OCR text, visual changes)
    """
    
    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()
        
        # Initialize feedback hub
        self.hub = FeedbackHub()
        self._setup_feedback_channels()
        
        # Initialize hardware interfaces
        self.cursor = KWinCursorPosition()
        self.windows = WindowManager()
        
        # State tracking
        self._last_pos = (0, 0)
        self._last_window = ""
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # MCP channel for accumulating events
        self.mcp_channel = self.hub.enable_mcp()
        
        # Emit init event
        self._emit("controller_init", {
            "config": asdict(self.config),
            "channels": list(self.hub.channels.keys())
        })
    
    def _setup_feedback_channels(self):
        """Set up configured feedback channels."""
        if self.config.enable_stdout:
            self.hub.enable_stdout()
        if self.config.enable_file:
            self.hub.enable_file(self.config.file_path)
        if self.config.enable_fifo:
            try:
                self.hub.enable_fifo(self.config.fifo_path)
            except:
                pass
        if self.config.enable_socket:
            try:
                self.hub.enable_socket(self.config.socket_path)
            except:
                pass
        if self.config.enable_agent_bus:
            self.hub.enable_agent_bus(self.config.agent_bus_state_dir)
    
    def _emit(self, event_type: str, data: Dict[str, Any]):
        """Emit feedback event."""
        self.hub.emit(event_type, data, source="live_controller")

    def _emit_pipeline(self, stage: str, data: Dict[str, Any]):
        if not self.config.enable_pipeline_trace:
            return
        payload = {"stage": stage, **data}
        self.hub.emit(f"pipeline.{stage}", payload, source="sensorimotor")
    
    def _get_pos(self) -> Tuple[int, int]:
        """Get cursor position."""
        return self.cursor.get_position()
    
    def _move(self, dx: int, dy: int):
        """Execute relative mouse move."""
        subprocess.run(["ydotool", "mousemove", "-x", str(dx), "-y", str(dy)], 
                      capture_output=True)
    
    def _click(self, button: str = "left"):
        """Click mouse button."""
        codes = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
        subprocess.run(["ydotool", "click", codes.get(button, "0xC0")], capture_output=True)
    
    def _at_edge(self, x: int, y: int) -> Optional[str]:
        """Check if at screen edge."""
        edges = []
        m = self.config.edge_margin
        if x <= m: edges.append("L")
        elif x >= self.config.screen_width - m: edges.append("R")
        if y <= m: edges.append("T")
        elif y >= self.config.screen_height - m: edges.append("B")
        return "".join(edges) if edges else None
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self._emit("monitoring_started", {})
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        self._emit("monitoring_stopped", {})
    
    def _monitor_loop(self):
        """Background loop tracking position and window changes."""
        pos_interval = 1.0 / self.config.position_update_hz
        win_check = 0
        
        while self._running:
            try:
                x, y = self._get_pos()
                
                # Position change?
                if abs(x - self._last_pos[0]) > 2 or abs(y - self._last_pos[1]) > 2:
                    edge = self._at_edge(x, y)
                    self._emit("cursor", {
                        "x": x, "y": y,
                        "dx": x - self._last_pos[0],
                        "dy": y - self._last_pos[1],
                        "edge": edge
                    })
                    self._emit_pipeline("see", {
                        "signal": "cursor",
                        "x": x,
                        "y": y,
                        "edge": edge,
                    })
                    self._last_pos = (x, y)
                
                # Window check (less frequent)
                win_check += 1
                if win_check >= int(self.config.position_update_hz / self.config.window_update_hz):
                    win_check = 0
                    try:
                        windows = self.windows.get_windows()
                        active = next((w for w in windows if w.active), None)
                        if active and active.caption != self._last_window:
                            self._emit("window_focus", {
                                "caption": active.caption[:60],
                                "class": active.resource_class
                            })
                            self._emit_pipeline("see", {
                                "signal": "window_focus",
                                "caption": active.caption[:60],
                                "class": active.resource_class,
                            })
                            self._last_window = active.caption
                    except:
                        pass
                
                time.sleep(pos_interval)
            except Exception as e:
                self._emit("monitor_error", {"error": str(e)})
                time.sleep(0.1)
    
    def move_to(self, target_x: int, target_y: int) -> Dict[str, Any]:
        """Move cursor to target with live feedback."""
        pipeline_id = uuid.uuid4().hex
        # Clamp target
        m = self.config.edge_margin
        target_x = max(m, min(target_x, self.config.screen_width - m - 1))
        target_y = max(m, min(target_y, self.config.screen_height - m - 1))
        
        start_x, start_y = self._get_pos()
        total_dist = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
        
        self._emit("move_start", {
            "from": [start_x, start_y],
            "to": [target_x, target_y],
            "distance": round(total_dist, 1),
            "pipeline_id": pipeline_id,
        })
        self._emit_pipeline("act", {
            "action": "move_to",
            "pipeline_id": pipeline_id,
            "target": [target_x, target_y],
        })
        
        start_time = time.time()
        iterations = 0
        path = [(start_x, start_y)]
        
        for i in range(self.config.mouse_max_iterations):
            iterations = i + 1
            x, y = self._get_pos()
            path.append((x, y))
            
            error_x = target_x - x
            error_y = target_y - y
            dist = math.sqrt(error_x**2 + error_y**2)
            edge = self._at_edge(x, y)
            
            # Emit position update
            self._emit("move_pos", {
                "p": [x, y],
                "err": round(dist, 1),
                "edge": edge,
                "iter": i,
                "pipeline_id": pipeline_id,
            })
            
            # Check convergence
            if dist <= self.config.mouse_tolerance:
                break
            
            # Edge handling
            if edge:
                if "L" in edge and error_x < 0: error_x = 0
                if "R" in edge and error_x > 0: error_x = 0
                if "T" in edge and error_y < 0: error_y = 0
                if "B" in edge and error_y > 0: error_y = 0
                
                dist = math.sqrt(error_x**2 + error_y**2)
                if dist <= self.config.mouse_tolerance:
                    break
            
            # Proportional control
            gain = 0.45 if dist > 80 else 0.35 if dist > 30 else 0.3
            dx = int(round(error_x * gain))
            dy = int(round(error_y * gain))
            
            # Human-like noise
            if self.config.mouse_human_like and dist > 40:
                dx += random.randint(-2, 2)
                dy += random.randint(-2, 2)
            
            # Minimum movement
            if dx == 0 and error_x != 0: dx = 1 if error_x > 0 else -1
            if dy == 0 and error_y != 0: dy = 1 if error_y > 0 else -1
            
            self._move(dx, dy)
            time.sleep(0.02)
        
        # Final position
        final_x, final_y = self._get_pos()
        final_dist = math.sqrt((target_x - final_x)**2 + (target_y - final_y)**2)
        elapsed_ms = (time.time() - start_time) * 1000
        success = final_dist <= self.config.mouse_tolerance
        
        result = {
            "success": success,
            "target": [target_x, target_y],
            "final": [final_x, final_y],
            "error": round(final_dist, 1),
            "iterations": iterations,
            "duration_ms": round(elapsed_ms, 1),
            "path_length": len(path),
            "pipeline_id": pipeline_id,
        }
        
        self._emit("move_done", result)
        self._emit_pipeline("verify", {
            "action": "move_to",
            "pipeline_id": pipeline_id,
            "success": success,
            "error": round(final_dist, 1),
        })
        return result
    
    def click_at(self, x: int, y: int, button: str = "left") -> Dict[str, Any]:
        """Move to position and click."""
        pipeline_id = uuid.uuid4().hex
        result = self.move_to(x, y)
        if result["success"]:
            self._click(button)
            self._emit("click", {"pos": [x, y], "button": button, "pipeline_id": pipeline_id})
            self._emit_pipeline("act", {
                "action": "click",
                "pipeline_id": pipeline_id,
                "pos": [x, y],
                "button": button,
            })
            result["clicked"] = True
        else:
            result["clicked"] = False
        return result
    
    def type_text(self, text: str, wpm: int = 180) -> Dict[str, Any]:
        """Type text with live progress feedback."""
        pipeline_id = uuid.uuid4().hex
        self._emit("type_start", {"length": len(text), "wpm": wpm, "pipeline_id": pipeline_id})
        self._emit_pipeline("act", {
            "action": "type_text",
            "pipeline_id": pipeline_id,
            "length": len(text),
        })
        
        cps = (wpm * 5) / 60
        delay = 1.0 / cps
        start_time = time.time()
        
        # Type character by character for progress tracking
        for i, char in enumerate(text):
            subprocess.run(["ydotool", "type", "--", char], capture_output=True)
            
            # Progress every 15 chars or on special chars
            if (i + 1) % 15 == 0 or char in '\n\t':
                self._emit("type_progress", {
                    "chars": i + 1,
                    "pct": round(100 * (i + 1) / len(text), 1),
                    "preview": text[max(0, i-10):i+1][-20:],
                    "pipeline_id": pipeline_id,
                })
            
            time.sleep(delay + random.uniform(-0.01, 0.015))
        
        elapsed_ms = (time.time() - start_time) * 1000
        actual_wpm = (len(text) / 5) / (elapsed_ms / 60000) if elapsed_ms > 0 else 0
        
        result = {
            "success": True,
            "chars_typed": len(text),
            "duration_ms": round(elapsed_ms, 1),
            "actual_wpm": round(actual_wpm, 1),
            "pipeline_id": pipeline_id,
        }
        self._emit("type_done", result)
        self._emit_pipeline("verify", {
            "action": "type_text",
            "pipeline_id": pipeline_id,
            "chars_typed": len(text),
        })
        return result
    
    def press_key(self, key: str) -> Dict[str, Any]:
        """Press a key."""
        pipeline_id = uuid.uuid4().hex
        key_codes = {
            "enter": "28:1 28:0",
            "tab": "15:1 15:0",
            "escape": "1:1 1:0",
            "backspace": "14:1 14:0",
            "space": "57:1 57:0",
            "up": "103:1 103:0",
            "down": "108:1 108:0",
            "left": "105:1 105:0",
            "right": "106:1 106:0",
        }
        
        code = key_codes.get(key.lower())
        if code:
            subprocess.run(["ydotool", "key"] + code.split(), capture_output=True)
            self._emit("key_press", {"key": key, "pipeline_id": pipeline_id})
            self._emit_pipeline("act", {"action": "press_key", "pipeline_id": pipeline_id, "key": key})
            return {"success": True, "key": key}
        else:
            self._emit("key_error", {"key": key, "error": "unknown key", "pipeline_id": pipeline_id})
            return {"success": False, "key": key, "error": "unknown key"}
    
    def get_state(self) -> Dict[str, Any]:
        """Get current controller state."""
        x, y = self._get_pos()
        
        try:
            windows = self.windows.get_windows()
            active = next((w for w in windows if w.active), None)
            active_caption = active.caption if active else None
        except:
            active_caption = None
        
        state = {
            "cursor": {"x": x, "y": y, "edge": self._at_edge(x, y)},
            "active_window": active_caption,
            "feedback_stats": self.hub.get_stats()
        }
        self._emit("state_query", state)
        self._emit_pipeline("see", {"signal": "state_query"})
        return state
    
    def get_mcp_events(self, clear: bool = True) -> Dict[str, Any]:
        """Get accumulated events in MCP format."""
        return self.mcp_channel.get_mcp_response(clear)
    
    def close(self):
        """Clean up resources."""
        self.stop_monitoring()
        self._emit("controller_close", {"final_stats": self.hub.get_stats()})
        self.hub.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def main():
    """CLI for testing the live controller."""
    import argparse
    parser = argparse.ArgumentParser(description="Live Controller")
    parser.add_argument("action", choices=["move", "click", "type", "key", "state", "demo", "monitor"])
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--text", type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--no-file", action="store_true", help="Disable file output")
    parser.add_argument("--agent-bus", action="store_true", help="Forward events to Agent Forge workspace bus")
    parser.add_argument("--agent-bus-dir", type=str, default="state", help="Agent Forge state dir")
    parser.add_argument("--pipeline-trace", action="store_true", help="Emit pipeline.see/act/verify events")
    args = parser.parse_args()
    
    config = ControllerConfig(
        enable_file=not args.no_file,
        enable_agent_bus=args.agent_bus,
        agent_bus_state_dir=args.agent_bus_dir,
        enable_pipeline_trace=args.pipeline_trace,
    )
    
    with LiveController(config) as ctrl:
        if args.action == "move" and args.x is not None and args.y is not None:
            ctrl.move_to(args.x, args.y)
        
        elif args.action == "click" and args.x is not None and args.y is not None:
            ctrl.click_at(args.x, args.y)
        
        elif args.action == "type" and args.text:
            ctrl.type_text(args.text)
        
        elif args.action == "key" and args.key:
            ctrl.press_key(args.key)
        
        elif args.action == "state":
            state = ctrl.get_state()
            print(json.dumps(state, indent=2))
        
        elif args.action == "monitor":
            ctrl.start_monitoring()
            print(f"# Monitoring for {args.duration}s... (events go to stdout and {config.file_path})")
            time.sleep(args.duration)
        
        elif args.action == "demo":
            ctrl._emit("demo_start", {"description": "Full capability demonstration"})
            
            # Move to center
            ctrl.move_to(960, 540)
            time.sleep(0.3)
            
            # Visit corners
            for name, pos in [("TL", (100, 100)), ("TR", (1820, 100)),
                             ("BR", (1820, 980)), ("BL", (100, 980))]:
                ctrl._emit("demo_corner", {"name": name})
                ctrl.move_to(pos[0], pos[1])
                time.sleep(0.2)
            
            # Back to center and type
            ctrl.move_to(960, 540)
            ctrl.click_at(960, 540)
            time.sleep(0.3)
            
            ctrl._emit("demo_end", {"message": "Demo complete"})


if __name__ == "__main__":
    main()
