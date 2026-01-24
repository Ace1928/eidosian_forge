#!/usr/bin/env python3
"""
Window Manager Perception via KWin Scripting API

Queries KDE Plasma window information without elevated permissions.
Uses KWin scripting via D-Bus for Wayland-native access.

Author: Eidos
Version: 1.0.0
"""

import subprocess
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .screen_state import BoundingBox


@dataclass
class WindowInfo:
    """Information about a window."""
    caption: str
    resource_class: str
    bbox: BoundingBox
    minimized: bool = False
    active: bool = False
    fullscreen: bool = False
    desktop: int = 1
    
    # Computed
    window_id: str = ""
    last_updated: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.window_id:
            # Generate ID from class and position
            self.window_id = f"{self.resource_class}_{self.bbox.x}_{self.bbox.y}"
    
    @property
    def center(self) -> Tuple[int, int]:
        return self.bbox.center
    
    def contains_point(self, x: int, y: int) -> bool:
        return self.bbox.contains(x, y)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "caption": self.caption,
            "resource_class": self.resource_class,
            "bbox": self.bbox.to_dict(),
            "minimized": self.minimized,
            "active": self.active,
            "fullscreen": self.fullscreen,
            "desktop": self.desktop,
            "window_id": self.window_id
        }


@dataclass
class WorkspaceInfo:
    """Information about the workspace/desktop."""
    display_width: int
    display_height: int
    cursor_pos: Tuple[int, int]
    current_desktop: int
    num_desktops: int
    active_window: Optional[str] = None


class KWinScriptRunner:
    """
    Executes KWin scripts and parses results from journal.
    
    Handles the full lifecycle: write script, load, run, parse, cleanup.
    """
    
    SCRIPT_DIR = Path("/tmp/eidos_kwin_scripts")
    OUTPUT_PREFIX = "EIDOS_KWIN:"
    
    def __init__(self):
        self.SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        self._script_counter = 0
    
    def _generate_script_name(self) -> str:
        self._script_counter += 1
        return f"eidos_script_{self._script_counter}_{int(time.time()*1000)}"
    
    def run_script(self, script_content: str, timeout: float = 2.0) -> List[str]:
        """
        Run a KWin script and return output lines.
        
        Args:
            script_content: JavaScript code to execute
            timeout: Max time to wait for output
            
        Returns:
            List of output lines (without prefix)
        """
        script_name = self._generate_script_name()
        script_path = self.SCRIPT_DIR / f"{script_name}.js"
        
        # Write script
        script_path.write_text(script_content)
        
        try:
            # Load script
            result = subprocess.run(
                ["qdbus", "org.kde.KWin", "/Scripting",
                 "org.kde.kwin.Scripting.loadScript",
                 str(script_path), script_name],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode != 0:
                return []
            
            # Start scripts
            subprocess.run(
                ["qdbus", "org.kde.KWin", "/Scripting",
                 "org.kde.kwin.Scripting.start"],
                capture_output=True, timeout=5
            )
            
            # Wait for execution
            time.sleep(0.2)
            
            # Read from journal
            output = self._read_journal_output(timeout)
            
            # Unload script
            subprocess.run(
                ["qdbus", "org.kde.KWin", "/Scripting",
                 "org.kde.kwin.Scripting.unloadScript", script_name],
                capture_output=True, timeout=5
            )
            
            return output
            
        finally:
            # Cleanup
            try:
                script_path.unlink(missing_ok=True)
            except:
                pass
    
    def _read_journal_output(self, timeout: float) -> List[str]:
        """Read script output from system journal."""
        try:
            result = subprocess.run(
                ["journalctl", "-n", "100", "--no-pager", "-o", "cat"],
                capture_output=True, text=True, timeout=timeout
            )
            
            lines = []
            for line in result.stdout.split('\n'):
                if self.OUTPUT_PREFIX in line:
                    # Extract content after prefix
                    idx = line.find(self.OUTPUT_PREFIX)
                    content = line[idx + len(self.OUTPUT_PREFIX):].strip()
                    lines.append(content)
            
            return lines
            
        except Exception:
            return []


class WindowManager:
    """
    Manages window perception using KWin scripting.
    
    Provides:
    - Window enumeration
    - Active window tracking
    - Workspace information
    - Window search by title/class
    """
    
    # KWin script for window enumeration
    WINDOW_LIST_SCRIPT = '''
var clients = workspace.clientList ? workspace.clientList() : 
              (workspace.stackingOrder || []);
              
print("EIDOS_KWIN:WORKSPACE:" + JSON.stringify({
    displayWidth: workspace.displayWidth,
    displayHeight: workspace.displayHeight,
    cursorX: workspace.cursorPos.x,
    cursorY: workspace.cursorPos.y,
    currentDesktop: workspace.currentDesktop,
    numDesktops: workspace.desktops.length
}));

for (var i = 0; i < clients.length; i++) {
    var w = clients[i];
    var geo = w.frameGeometry || {x: w.x, y: w.y, width: w.width, height: w.height};
    print("EIDOS_KWIN:WINDOW:" + JSON.stringify({
        caption: w.caption || "",
        resourceClass: w.resourceClass || "",
        x: geo.x,
        y: geo.y,
        width: geo.width,
        height: geo.height,
        minimized: w.minimized || false,
        active: w.active || false,
        fullScreen: w.fullScreen || false,
        desktop: w.desktop || 1
    }));
}
print("EIDOS_KWIN:DONE");
'''
    
    # Script to get active window only
    ACTIVE_WINDOW_SCRIPT = '''
var w = workspace.activeClient;
if (w) {
    var geo = w.frameGeometry || {x: w.x, y: w.y, width: w.width, height: w.height};
    print("EIDOS_KWIN:ACTIVE:" + JSON.stringify({
        caption: w.caption || "",
        resourceClass: w.resourceClass || "",
        x: geo.x,
        y: geo.y,
        width: geo.width,
        height: geo.height
    }));
} else {
    print("EIDOS_KWIN:ACTIVE:null");
}
'''
    
    def __init__(self):
        self.runner = KWinScriptRunner()
        
        # Cached state
        self._windows: List[WindowInfo] = []
        self._workspace: Optional[WorkspaceInfo] = None
        self._last_update: float = 0
        self._cache_ttl: float = 0.5  # Cache for 500ms
        
        # Statistics
        self.stats = {
            "queries": 0,
            "cache_hits": 0
        }
    
    def _should_update_cache(self) -> bool:
        return time.time() - self._last_update > self._cache_ttl
    
    def get_windows(self, force_refresh: bool = False) -> List[WindowInfo]:
        """
        Get list of all windows.
        
        Args:
            force_refresh: Bypass cache
        """
        if not force_refresh and not self._should_update_cache():
            self.stats["cache_hits"] += 1
            return self._windows
        
        self.stats["queries"] += 1
        self._update_state()
        return self._windows
    
    def get_workspace_info(self, force_refresh: bool = False) -> Optional[WorkspaceInfo]:
        """Get workspace information."""
        if not force_refresh and not self._should_update_cache():
            return self._workspace
        
        self._update_state()
        return self._workspace
    
    def _update_state(self):
        """Query KWin and update cached state."""
        output = self.runner.run_script(self.WINDOW_LIST_SCRIPT)
        
        windows = []
        workspace = None
        
        for line in output:
            if line.startswith("WORKSPACE:"):
                try:
                    data = json.loads(line[10:])
                    workspace = WorkspaceInfo(
                        display_width=data.get("displayWidth", 1920),
                        display_height=data.get("displayHeight", 1080),
                        cursor_pos=(data.get("cursorX", 0), data.get("cursorY", 0)),
                        current_desktop=data.get("currentDesktop", 1),
                        num_desktops=data.get("numDesktops", 1)
                    )
                except:
                    pass
                    
            elif line.startswith("WINDOW:"):
                try:
                    data = json.loads(line[7:])
                    window = WindowInfo(
                        caption=data.get("caption", ""),
                        resource_class=data.get("resourceClass", ""),
                        bbox=BoundingBox(
                            x=data.get("x", 0),
                            y=data.get("y", 0),
                            width=data.get("width", 0),
                            height=data.get("height", 0)
                        ),
                        minimized=data.get("minimized", False),
                        active=data.get("active", False),
                        fullscreen=data.get("fullScreen", False),
                        desktop=data.get("desktop", 1)
                    )
                    
                    # Track active window
                    if window.active and workspace:
                        workspace.active_window = window.window_id
                    
                    windows.append(window)
                except:
                    pass
        
        self._windows = windows
        self._workspace = workspace
        self._last_update = time.time()
    
    def get_active_window(self) -> Optional[WindowInfo]:
        """Get currently active window."""
        windows = self.get_windows()
        for w in windows:
            if w.active:
                return w
        return None
    
    def find_window_by_title(self, title: str, 
                              partial: bool = True) -> Optional[WindowInfo]:
        """Find window by title."""
        windows = self.get_windows()
        title_lower = title.lower()
        
        for w in windows:
            caption_lower = w.caption.lower()
            if partial:
                if title_lower in caption_lower:
                    return w
            else:
                if title_lower == caption_lower:
                    return w
        
        return None
    
    def find_window_by_class(self, resource_class: str) -> Optional[WindowInfo]:
        """Find window by resource class."""
        windows = self.get_windows()
        class_lower = resource_class.lower()
        
        for w in windows:
            if class_lower in w.resource_class.lower():
                return w
        
        return None
    
    def find_windows_at_point(self, x: int, y: int) -> List[WindowInfo]:
        """Find all windows containing a point."""
        windows = self.get_windows()
        return [w for w in windows if w.contains_point(x, y)]
    
    def get_window_at_cursor(self) -> Optional[WindowInfo]:
        """Get window under cursor."""
        workspace = self.get_workspace_info()
        if not workspace:
            return None
        
        windows = self.find_windows_at_point(*workspace.cursor_pos)
        # Return topmost (last in stacking order, usually last in list)
        return windows[-1] if windows else None
    
    def get_visible_windows(self) -> List[WindowInfo]:
        """Get non-minimized windows on current desktop."""
        windows = self.get_windows()
        workspace = self.get_workspace_info()
        current_desktop = workspace.current_desktop if workspace else 1
        
        return [
            w for w in windows 
            if not w.minimized and 
               (w.desktop == current_desktop or w.desktop == -1)  # -1 = all desktops
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get window manager statistics."""
        return {
            **self.stats,
            "cached_windows": len(self._windows),
            "cache_age_ms": (time.time() - self._last_update) * 1000
        }
