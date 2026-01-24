#!/usr/bin/env python3
"""
Unified Perception System

Integrates all perception modalities into a coherent world model:
- Visual perception (screenshots, change detection)
- OCR (text recognition)
- Window management (KWin)
- Cursor tracking
- Element database with spatial indexing

Author: Eidos
Version: 1.0.0
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

from .screen_state import (
    BoundingBox, PerceivedElement, ScreenFrame, 
    ChangeRegion, ScreenCapture, ChangeDetector, ScreenPerception
)
from .ocr_engine import OCREngine, TextRegion
from .window_manager import WindowManager, WindowInfo, WorkspaceInfo


@dataclass
class PerceptionConfig:
    """Configuration for perception system."""
    # Update rates
    visual_update_hz: float = 2.0      # Screenshot rate
    window_update_hz: float = 1.0      # Window list rate
    ocr_on_change_only: bool = True    # Only OCR changed regions
    
    # Change detection
    change_threshold: int = 30         # Pixel difference threshold
    min_change_region: int = 100       # Minimum pixels for change
    
    # Element tracking
    element_stale_time: float = 30.0   # Remove elements not seen for this long
    
    # Performance
    max_ocr_regions_per_frame: int = 3
    cache_size: int = 50


@dataclass
class PerceptionState:
    """Current state of perception system."""
    timestamp: float
    
    # Screen
    screen_width: int
    screen_height: int
    frame_hash: Optional[str]
    
    # Cursor
    cursor_position: Tuple[int, int]
    cursor_at_edge: Dict[str, bool]
    
    # Windows
    windows: List[WindowInfo]
    active_window: Optional[WindowInfo]
    window_under_cursor: Optional[WindowInfo]
    
    # Text (OCR)
    visible_text_regions: List[TextRegion]
    
    # Changes
    recent_changes: List[ChangeRegion]
    frame_changed: bool
    
    # Elements
    tracked_elements: Dict[str, PerceivedElement]


class SpatialIndex:
    """
    Simple spatial indexing for fast point-in-region queries.
    Uses a grid-based approach for O(1) average lookup.
    """
    
    def __init__(self, width: int, height: int, cell_size: int = 100):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        self.cols = (width + cell_size - 1) // cell_size
        self.rows = (height + cell_size - 1) // cell_size
        
        # Grid cells contain element IDs
        self.grid: Dict[Tuple[int, int], set] = defaultdict(set)
        self.elements: Dict[str, BoundingBox] = {}
    
    def _get_cells(self, bbox: BoundingBox) -> List[Tuple[int, int]]:
        """Get grid cells overlapping with bbox."""
        cells = []
        
        start_col = max(0, bbox.x // self.cell_size)
        end_col = min(self.cols - 1, bbox.x2 // self.cell_size)
        start_row = max(0, bbox.y // self.cell_size)
        end_row = min(self.rows - 1, bbox.y2 // self.cell_size)
        
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                cells.append((col, row))
        
        return cells
    
    def insert(self, element_id: str, bbox: BoundingBox):
        """Insert element into spatial index."""
        self.elements[element_id] = bbox
        for cell in self._get_cells(bbox):
            self.grid[cell].add(element_id)
    
    def remove(self, element_id: str):
        """Remove element from spatial index."""
        if element_id not in self.elements:
            return
        
        bbox = self.elements[element_id]
        for cell in self._get_cells(bbox):
            self.grid[cell].discard(element_id)
        
        del self.elements[element_id]
    
    def query_point(self, x: int, y: int) -> List[str]:
        """Get elements containing point."""
        cell = (x // self.cell_size, y // self.cell_size)
        candidates = self.grid.get(cell, set())
        
        return [
            eid for eid in candidates
            if self.elements[eid].contains(x, y)
        ]
    
    def query_region(self, bbox: BoundingBox) -> List[str]:
        """Get elements overlapping with region."""
        candidates = set()
        
        for cell in self._get_cells(bbox):
            candidates.update(self.grid.get(cell, set()))
        
        return [
            eid for eid in candidates
            if self.elements[eid].overlaps(bbox)
        ]
    
    def clear(self):
        """Clear the index."""
        self.grid.clear()
        self.elements.clear()


class UnifiedPerception:
    """
    Main perception system coordinating all modalities.
    
    Provides:
    - Real-time screen state tracking
    - Multi-modal element detection
    - Spatial queries
    - Change-driven updates for efficiency
    """
    
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        
        # Initialize subsystems
        self.screen = ScreenPerception()
        self.ocr = OCREngine(cache_size=self.config.cache_size)
        self.windows = WindowManager()
        
        # Spatial index
        ws = self.windows.get_workspace_info()
        width = ws.display_width if ws else 1920
        height = ws.display_height if ws else 1080
        self.spatial_index = SpatialIndex(width, height)
        
        # Element database
        self.elements: Dict[str, PerceivedElement] = {}
        self._element_counter = 0
        
        # Text regions (indexed by position hash)
        self.text_regions: Dict[str, TextRegion] = {}
        
        # Timing
        self._last_visual_update = 0
        self._last_window_update = 0
        self._last_ocr_update = 0
        
        # Callbacks
        self._change_callbacks: List[Callable] = []
        
        # Background thread (optional)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            "updates": 0,
            "visual_updates": 0,
            "window_updates": 0,
            "ocr_updates": 0,
            "elements_tracked": 0,
            "text_regions": 0
        }
    
    def update(self) -> PerceptionState:
        """
        Perform a perception update cycle.
        
        Returns current perception state.
        """
        now = time.time()
        self.stats["updates"] += 1
        
        # Visual update
        frame_changed = False
        recent_changes = []
        
        visual_interval = 1.0 / self.config.visual_update_hz
        if now - self._last_visual_update >= visual_interval:
            result = self.screen.update()
            frame_changed = result.get("frame_changed", False)
            recent_changes = result.get("changes", [])
            self._last_visual_update = now
            self.stats["visual_updates"] += 1
            
            # OCR on changes
            if frame_changed and self.config.ocr_on_change_only:
                self._ocr_changed_regions(recent_changes)
        
        # Window update
        window_interval = 1.0 / self.config.window_update_hz
        if now - self._last_window_update >= window_interval:
            self._update_windows()
            self._last_window_update = now
            self.stats["window_updates"] += 1
        
        # Get cursor from windows subsystem
        ws = self.windows.get_workspace_info()
        if ws:
            self.screen.update_cursor(*ws.cursor_pos)
        
        # Prune stale elements
        self._prune_stale_elements()
        
        # Build state
        return self._build_state(frame_changed, recent_changes)
    
    def _ocr_changed_regions(self, changes: List[ChangeRegion]):
        """Run OCR on changed regions."""
        if not self.screen.current_frame:
            return
        
        # Limit OCR calls
        changes_to_process = sorted(
            changes, 
            key=lambda c: c.pixel_count, 
            reverse=True
        )[:self.config.max_ocr_regions_per_frame]
        
        for change in changes_to_process:
            result = self.ocr.ocr_frame(
                self.screen.current_frame,
                change.bbox
            )
            
            if result.success:
                for tr in result.text_regions:
                    self._add_text_region(tr)
        
        self.stats["ocr_updates"] += 1
    
    def _add_text_region(self, tr: TextRegion):
        """Add text region to tracking."""
        # Use position as key for deduplication
        key = f"text_{tr.bbox.x}_{tr.bbox.y}_{hash(tr.text) % 10000}"
        self.text_regions[key] = tr
        self.stats["text_regions"] = len(self.text_regions)
    
    def _update_windows(self):
        """Update window tracking."""
        windows = self.windows.get_windows(force_refresh=True)
        
        # Add/update window elements
        for w in windows:
            elem_id = f"window_{w.window_id}"
            
            if elem_id in self.elements:
                # Update existing
                elem = self.elements[elem_id]
                elem.bbox = w.bbox
                elem.text = w.caption
                elem.last_seen = time.time()
                elem.stable_frames += 1
                elem.properties["active"] = w.active
                elem.properties["minimized"] = w.minimized
            else:
                # Add new
                elem = PerceivedElement(
                    element_id=elem_id,
                    element_type="window",
                    bbox=w.bbox,
                    text=w.caption,
                    properties={
                        "resource_class": w.resource_class,
                        "active": w.active,
                        "minimized": w.minimized
                    }
                )
                self.elements[elem_id] = elem
                self.spatial_index.insert(elem_id, w.bbox)
        
        self.stats["elements_tracked"] = len(self.elements)
    
    def _prune_stale_elements(self):
        """Remove elements not seen recently."""
        now = time.time()
        stale = [
            eid for eid, elem in self.elements.items()
            if elem.time_since_seen() > self.config.element_stale_time
        ]
        
        for eid in stale:
            self.spatial_index.remove(eid)
            del self.elements[eid]
    
    def _build_state(self, frame_changed: bool, 
                     changes: List[ChangeRegion]) -> PerceptionState:
        """Build current perception state."""
        ws = self.windows.get_workspace_info()
        windows = self.windows.get_windows()
        active = self.windows.get_active_window()
        under_cursor = self.windows.get_window_at_cursor()
        
        return PerceptionState(
            timestamp=time.time(),
            screen_width=ws.display_width if ws else 1920,
            screen_height=ws.display_height if ws else 1080,
            frame_hash=self.screen.current_frame.hash if self.screen.current_frame else None,
            cursor_position=self.screen.cursor_position,
            cursor_at_edge=self.screen.cursor_at_edge,
            windows=windows,
            active_window=active,
            window_under_cursor=under_cursor,
            visible_text_regions=list(self.text_regions.values()),
            recent_changes=changes,
            frame_changed=frame_changed,
            tracked_elements=dict(self.elements)
        )
    
    # === Query Methods ===
    
    def get_element_at(self, x: int, y: int) -> Optional[PerceivedElement]:
        """Get element at position."""
        elem_ids = self.spatial_index.query_point(x, y)
        if elem_ids:
            return self.elements.get(elem_ids[0])
        return None
    
    def get_elements_in_region(self, bbox: BoundingBox) -> List[PerceivedElement]:
        """Get elements in region."""
        elem_ids = self.spatial_index.query_region(bbox)
        return [self.elements[eid] for eid in elem_ids if eid in self.elements]
    
    def find_text_on_screen(self, query: str, 
                            refresh_ocr: bool = False) -> List[TextRegion]:
        """Find text on screen."""
        if refresh_ocr and self.screen.current_frame:
            result = self.ocr.ocr_frame(self.screen.current_frame)
            if result.success:
                for tr in result.text_regions:
                    self._add_text_region(tr)
        
        query_lower = query.lower()
        return [
            tr for tr in self.text_regions.values()
            if query_lower in tr.text.lower()
        ]
    
    def find_text_location(self, query: str, 
                           refresh_ocr: bool = True) -> Optional[Tuple[int, int]]:
        """Find location of text on screen."""
        matches = self.find_text_on_screen(query, refresh_ocr)
        if matches:
            return matches[0].bbox.center
        return None
    
    def find_window(self, title: str) -> Optional[WindowInfo]:
        """Find window by title."""
        return self.windows.find_window_by_title(title)
    
    def get_cursor_context(self) -> Dict[str, Any]:
        """Get context around cursor."""
        ws = self.windows.get_workspace_info()
        if not ws:
            return {}
        
        x, y = ws.cursor_pos
        
        return {
            "position": (x, y),
            "at_edge": self.screen.cursor_at_edge,
            "window": self.windows.get_window_at_cursor(),
            "elements": [
                self.elements[eid] 
                for eid in self.spatial_index.query_point(x, y)
                if eid in self.elements
            ],
            "nearby_text": [
                tr for tr in self.text_regions.values()
                if abs(tr.bbox.center[0] - x) < 200 and abs(tr.bbox.center[1] - y) < 200
            ]
        }
    
    def is_cursor_at_edge(self) -> bool:
        """Check if cursor is at screen edge."""
        return self.screen.is_at_edge()
    
    def get_edge_direction(self) -> Optional[Tuple[int, int]]:
        """Get direction to move away from edge."""
        return self.screen.get_edge_direction()
    
    # === Callbacks ===
    
    def on_change(self, callback: Callable[[PerceptionState], None]):
        """Register callback for state changes."""
        self._change_callbacks.append(callback)
    
    def _notify_change(self, state: PerceptionState):
        """Notify registered callbacks."""
        for cb in self._change_callbacks:
            try:
                cb(state)
            except Exception:
                pass
    
    # === Background Thread ===
    
    def start_background(self):
        """Start background perception updates."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()
    
    def stop_background(self):
        """Stop background updates."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def _background_loop(self):
        """Background update loop."""
        while self._running:
            try:
                state = self.update()
                if state.frame_changed:
                    self._notify_change(state)
            except Exception:
                pass
            
            time.sleep(0.1)
    
    # === Stats ===
    
    def get_stats(self) -> Dict[str, Any]:
        """Get perception statistics."""
        return {
            **self.stats,
            "screen_stats": self.screen.get_state_summary(),
            "ocr_stats": self.ocr.get_stats(),
            "window_stats": self.windows.get_stats()
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_background()
        self.screen.cleanup()
        self.ocr.clear_cache()


# Convenience function
def create_perception() -> UnifiedPerception:
    """Create and return a perception system."""
    return UnifiedPerception()
