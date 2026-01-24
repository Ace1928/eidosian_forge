#!/usr/bin/env python3
"""
Screen State Perception Engine

Efficiently tracks screen state changes using delta detection.
Maintains a model of visible elements (windows, text, icons).

Author: Eidos
Version: 1.0.0
"""

import subprocess
import os
import time
import hashlib
from typing import Tuple, Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from PIL import Image
import numpy as np
from collections import deque


@dataclass
class BoundingBox:
    """A rectangular region on screen."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.x2 and self.y <= y < self.y2
    
    def overlaps(self, other: 'BoundingBox') -> bool:
        return not (self.x2 <= other.x or other.x2 <= self.x or
                   self.y2 <= other.y or other.y2 <= self.y)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x2, self.y2)
    
    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class PerceivedElement:
    """An element detected on screen."""
    element_id: str
    element_type: str  # "window", "text", "icon", "button", "region"
    bbox: BoundingBox
    confidence: float = 1.0
    text: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)
    stable_frames: int = 0
    
    def age(self) -> float:
        return time.time() - self.first_seen
    
    def time_since_seen(self) -> float:
        return time.time() - self.last_seen


@dataclass
class ScreenFrame:
    """A captured screen frame with metadata."""
    timestamp: float
    image: Optional[Image.Image]
    path: Optional[str]
    width: int
    height: int
    hash: str
    
    # Computed on demand
    _array: Optional[np.ndarray] = field(default=None, repr=False)
    _grayscale: Optional[np.ndarray] = field(default=None, repr=False)
    
    @property
    def array(self) -> np.ndarray:
        if self._array is None and self.image:
            self._array = np.array(self.image)
        return self._array
    
    @property
    def grayscale(self) -> np.ndarray:
        if self._grayscale is None and self.image:
            self._grayscale = np.array(self.image.convert('L'))
        return self._grayscale


@dataclass
class ChangeRegion:
    """A region where change was detected."""
    bbox: BoundingBox
    change_magnitude: float  # 0-1
    pixel_count: int
    timestamp: float = field(default_factory=time.time)


class ScreenCapture:
    """Efficient screen capture with caching."""
    
    def __init__(self, cache_dir: str = "/tmp/eidos_perception"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_capture: Optional[ScreenFrame] = None
        self._capture_count = 0
    
    def capture(self, reuse_if_recent_ms: int = 50) -> Optional[ScreenFrame]:
        """
        Capture screen, optionally reusing recent capture.
        
        Args:
            reuse_if_recent_ms: Reuse last capture if within this time
        """
        now = time.time()
        
        # Reuse recent capture
        if (self._last_capture and 
            (now - self._last_capture.timestamp) * 1000 < reuse_if_recent_ms):
            return self._last_capture
        
        # Take new screenshot
        self._capture_count += 1
        filename = f"frame_{self._capture_count:06d}.png"
        filepath = self.cache_dir / filename
        
        try:
            result = subprocess.run(
                ["spectacle", "-b", "-n", "-o", str(filepath)],
                capture_output=True, timeout=5
            )
            
            if result.returncode != 0 or not filepath.exists():
                return None
            
            img = Image.open(filepath)
            
            # Compute perceptual hash
            small = img.resize((16, 16)).convert('L')
            pixels = np.array(small)
            avg = pixels.mean()
            bits = (pixels > avg).flatten()
            hash_val = ''.join('1' if b else '0' for b in bits)
            
            frame = ScreenFrame(
                timestamp=now,
                image=img.copy(),
                path=str(filepath),
                width=img.width,
                height=img.height,
                hash=hash_val
            )
            
            # Cleanup old file
            if self._last_capture and self._last_capture.path:
                try:
                    Path(self._last_capture.path).unlink(missing_ok=True)
                except:
                    pass
            
            self._last_capture = frame
            return frame
            
        except Exception as e:
            return None
    
    def cleanup(self):
        """Remove cached files."""
        for f in self.cache_dir.glob("frame_*.png"):
            try:
                f.unlink()
            except:
                pass


class ChangeDetector:
    """Detects changes between screen frames."""
    
    def __init__(self, threshold: int = 30, min_region_size: int = 100):
        self.threshold = threshold  # Pixel difference threshold
        self.min_region_size = min_region_size  # Min pixels for a change region
    
    def detect_changes(self, frame1: ScreenFrame, frame2: ScreenFrame) -> List[ChangeRegion]:
        """
        Detect changed regions between two frames.
        Uses efficient numpy operations.
        """
        if frame1.hash == frame2.hash:
            return []  # Identical frames
        
        # Get grayscale arrays
        arr1 = frame1.grayscale
        arr2 = frame2.grayscale
        
        if arr1.shape != arr2.shape:
            return []  # Different sizes
        
        # Compute absolute difference
        diff = np.abs(arr1.astype(np.int16) - arr2.astype(np.int16))
        
        # Threshold
        changed = diff > self.threshold
        
        # Find changed regions using connected components
        regions = self._find_regions(changed)
        
        # Convert to ChangeRegion objects
        change_regions = []
        for bbox, magnitude, count in regions:
            if count >= self.min_region_size:
                change_regions.append(ChangeRegion(
                    bbox=bbox,
                    change_magnitude=magnitude,
                    pixel_count=count
                ))
        
        return change_regions
    
    def _find_regions(self, mask: np.ndarray) -> List[Tuple[BoundingBox, float, int]]:
        """Find bounding boxes of changed regions."""
        # Simple approach: find non-zero bounds
        # For production, use scipy.ndimage.label for connected components
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return []
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        region = mask[rmin:rmax+1, cmin:cmax+1]
        count = region.sum()
        magnitude = count / region.size if region.size > 0 else 0
        
        bbox = BoundingBox(
            x=int(cmin), y=int(rmin),
            width=int(cmax - cmin + 1),
            height=int(rmax - rmin + 1)
        )
        
        return [(bbox, magnitude, int(count))]
    
    def compute_similarity(self, frame1: ScreenFrame, frame2: ScreenFrame) -> float:
        """Compute similarity between frames (0-1, 1=identical)."""
        if frame1.hash == frame2.hash:
            return 1.0
        
        # Hamming distance of perceptual hashes
        h1 = frame1.hash
        h2 = frame2.hash
        
        if len(h1) != len(h2):
            return 0.0
        
        diff = sum(c1 != c2 for c1, c2 in zip(h1, h2))
        return 1.0 - (diff / len(h1))


class ScreenPerception:
    """
    Main perception engine - maintains model of screen state.
    
    Tracks:
    - Current screen state
    - Changed regions
    - Detected elements (windows, text, icons)
    - Cursor position and movement
    """
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.capture = ScreenCapture()
        self.change_detector = ChangeDetector()
        
        # State
        self.current_frame: Optional[ScreenFrame] = None
        self.previous_frame: Optional[ScreenFrame] = None
        
        # Element tracking
        self.elements: Dict[str, PerceivedElement] = {}
        self._element_counter = 0
        
        # Change history (for temporal analysis)
        self.change_history: deque = deque(maxlen=100)
        
        # Cursor state
        self.cursor_position: Tuple[int, int] = (0, 0)
        self.cursor_at_edge: Dict[str, bool] = {
            "left": False, "right": False, "top": False, "bottom": False
        }
        
        # Screen edge margins
        self.edge_margin = 5
        
        # Statistics
        self.stats = {
            "frames_captured": 0,
            "changes_detected": 0,
            "elements_tracked": 0
        }
    
    def update(self) -> Dict[str, Any]:
        """
        Update perception state.
        
        Returns dict with:
        - frame_changed: bool
        - changes: List[ChangeRegion]
        - cursor_at_edge: Dict[str, bool]
        """
        result = {
            "frame_changed": False,
            "changes": [],
            "cursor_at_edge": self.cursor_at_edge.copy(),
            "timestamp": time.time()
        }
        
        # Capture new frame
        new_frame = self.capture.capture()
        if not new_frame:
            return result
        
        self.stats["frames_captured"] += 1
        
        # Compare with previous
        if self.current_frame:
            self.previous_frame = self.current_frame
            
            # Detect changes
            changes = self.change_detector.detect_changes(
                self.previous_frame, new_frame
            )
            
            if changes:
                result["frame_changed"] = True
                result["changes"] = changes
                self.change_history.extend(changes)
                self.stats["changes_detected"] += len(changes)
        
        self.current_frame = new_frame
        return result
    
    def update_cursor(self, x: int, y: int):
        """Update cursor position and edge detection."""
        self.cursor_position = (x, y)
        
        # Detect edge contact
        self.cursor_at_edge = {
            "left": x <= self.edge_margin,
            "right": x >= self.screen_width - self.edge_margin,
            "top": y <= self.edge_margin,
            "bottom": y >= self.screen_height - self.edge_margin
        }
    
    def is_at_edge(self) -> bool:
        """Check if cursor is at any screen edge."""
        return any(self.cursor_at_edge.values())
    
    def get_edge_direction(self) -> Optional[Tuple[int, int]]:
        """Get direction away from edge if at edge."""
        dx, dy = 0, 0
        
        if self.cursor_at_edge["left"]:
            dx = 1
        elif self.cursor_at_edge["right"]:
            dx = -1
        
        if self.cursor_at_edge["top"]:
            dy = 1
        elif self.cursor_at_edge["bottom"]:
            dy = -1
        
        return (dx, dy) if dx != 0 or dy != 0 else None
    
    def add_element(self, element_type: str, bbox: BoundingBox,
                    text: Optional[str] = None, **properties) -> str:
        """Add or update a perceived element."""
        self._element_counter += 1
        element_id = f"{element_type}_{self._element_counter}"
        
        self.elements[element_id] = PerceivedElement(
            element_id=element_id,
            element_type=element_type,
            bbox=bbox,
            text=text,
            properties=properties
        )
        
        self.stats["elements_tracked"] = len(self.elements)
        return element_id
    
    def get_element_at(self, x: int, y: int) -> Optional[PerceivedElement]:
        """Get element at position."""
        for elem in self.elements.values():
            if elem.bbox.contains(x, y):
                return elem
        return None
    
    def get_elements_in_region(self, bbox: BoundingBox) -> List[PerceivedElement]:
        """Get all elements overlapping with region."""
        return [e for e in self.elements.values() if e.bbox.overlaps(bbox)]
    
    def prune_stale_elements(self, max_age: float = 30.0):
        """Remove elements not seen recently."""
        now = time.time()
        stale = [eid for eid, e in self.elements.items() 
                 if e.time_since_seen() > max_age]
        for eid in stale:
            del self.elements[eid]
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current perception state summary."""
        return {
            "timestamp": time.time(),
            "screen_size": (self.screen_width, self.screen_height),
            "cursor_position": self.cursor_position,
            "cursor_at_edge": self.cursor_at_edge,
            "frame_hash": self.current_frame.hash if self.current_frame else None,
            "element_count": len(self.elements),
            "recent_changes": len(self.change_history),
            "stats": self.stats.copy()
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.capture.cleanup()
