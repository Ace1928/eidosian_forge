"""Frame processing pipeline for high-performance streaming.

This module provides the frame processing pipeline including edge detection,
color extraction, and parallel processing for optimal performance.

Classes:
    FrameProcessor: High-performance frame processing pipeline
    
Functions:
    supersample_image: Supersample image for better quality
    rgb_to_gray: Convert RGB to grayscale
    detect_edges: Multi-algorithm edge detection
"""
from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .types import EdgeDetector, GradientResult, QualityLevel


# ═══════════════════════════════════════════════════════════════
# Image Processing Utilities
# ═══════════════════════════════════════════════════════════════

def supersample_image(image: "Image.Image", scale_factor: int) -> "Image.Image":
    """Supersample image for improved quality.
    
    Args:
        image: PIL Image to supersample
        scale_factor: Upscale factor (1-4)
        
    Returns:
        Supersampled image
    """
    if scale_factor <= 1:
        return image
    
    new_size = (image.width * scale_factor, image.height * scale_factor)
    return image.resize(new_size, Image.Resampling.LANCZOS)


def rgb_to_gray(image_array: np.ndarray) -> np.ndarray:
    """Convert RGB array to grayscale with perceptual weighting.
    
    Uses ITU-R BT.601 luma coefficients for perceptually accurate
    grayscale conversion.
    
    Args:
        image_array: RGB image array (H, W, 3)
        
    Returns:
        Grayscale array (H, W)
    """
    if len(image_array.shape) == 2:
        return image_array
    
    # ITU-R BT.601 luma coefficients
    return np.dot(image_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def detect_edges(
    gray_array: np.ndarray,
    algorithm: str = "sobel",
    threshold: int = 50,
) -> GradientResult:
    """Detect edges using specified algorithm.
    
    Args:
        gray_array: Grayscale image array
        algorithm: Detection algorithm (sobel, prewitt, scharr, laplacian, canny)
        threshold: Edge detection sensitivity (0-255)
        
    Returns:
        GradientResult with magnitude and direction information
    """
    if not HAS_OPENCV:
        # Fallback: return uniform result
        h, w = gray_array.shape[:2]
        return {
            "magnitude": np.zeros((h, w), dtype=np.uint8),
            "gradient_x": np.zeros((h, w), dtype=np.float32),
            "gradient_y": np.zeros((h, w), dtype=np.float32),
            "direction": np.zeros((h, w), dtype=np.float32),
        }
    
    algorithm = algorithm.lower()
    gray_float = gray_array.astype(np.float32)
    
    if algorithm == "canny":
        edges = cv2.Canny(gray_array, threshold, threshold * 2)
        return {
            "magnitude": edges,
            "gradient_x": np.zeros_like(gray_float),
            "gradient_y": np.zeros_like(gray_float),
            "direction": np.zeros_like(gray_float),
        }
    
    elif algorithm == "laplacian":
        lap = cv2.Laplacian(gray_float, cv2.CV_32F)
        magnitude = np.abs(lap)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        return {
            "magnitude": magnitude,
            "gradient_x": lap,
            "gradient_y": lap,
            "direction": np.zeros_like(gray_float),
        }
    
    elif algorithm == "prewitt":
        # Prewitt kernels
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        grad_x = cv2.filter2D(gray_float, cv2.CV_32F, kernel_x)
        grad_y = cv2.filter2D(gray_float, cv2.CV_32F, kernel_y)
    
    elif algorithm == "scharr":
        grad_x = cv2.Scharr(gray_float, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(gray_float, cv2.CV_32F, 0, 1)
    
    else:  # sobel (default)
        grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    
    # Calculate magnitude and direction
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    direction = np.arctan2(grad_y, grad_x)
    
    return {
        "magnitude": magnitude,
        "gradient_x": grad_x,
        "gradient_y": grad_y,
        "direction": direction,
    }


# ═══════════════════════════════════════════════════════════════
# Frame Processor
# ═══════════════════════════════════════════════════════════════

class FrameProcessor:
    """High-performance frame processing pipeline.
    
    Provides parallel frame processing with configurable quality
    settings and edge detection algorithms.
    
    Attributes:
        scale_factor: Supersampling scale (1-4)
        block_width: Character block width in pixels
        block_height: Character block height in pixels
        edge_threshold: Edge detection sensitivity
        algorithm: Edge detection algorithm
        color_enabled: Whether to extract color
        max_workers: Thread pool size
    """
    
    def __init__(
        self,
        scale_factor: int = 1,
        block_width: int = 2,
        block_height: int = 4,
        edge_threshold: int = 50,
        algorithm: str = "sobel",
        color_enabled: bool = True,
        max_workers: Optional[int] = None,
    ):
        """Initialize frame processor.
        
        Args:
            scale_factor: Supersampling scale
            block_width: Pixels per character width
            block_height: Pixels per character height
            edge_threshold: Edge detection threshold
            algorithm: Edge detection algorithm
            color_enabled: Enable color extraction
            max_workers: Thread pool size (None=auto)
        """
        self.scale_factor = max(1, min(4, scale_factor))
        self.block_width = max(1, block_width)
        self.block_height = max(1, block_height)
        self.edge_threshold = max(0, min(255, edge_threshold))
        self.algorithm = algorithm
        self.color_enabled = color_enabled
        
        # Thread pool for parallel processing
        import os
        cpu_count = os.cpu_count() or 4
        self.max_workers = max_workers or min(8, cpu_count)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="frame_proc",
        )
        
        # Processing caches
        self._lock = threading.RLock()
    
    def process_frame(
        self,
        frame: np.ndarray,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process a single frame for glyph rendering.
        
        Args:
            frame: BGR frame from video capture
            target_width: Target character width (None=auto)
            target_height: Target character height (None=auto)
            
        Returns:
            Dictionary with processed data:
                - blocks: 2D array of block data
                - colors: RGB colors for each block (if color_enabled)
                - edges: Edge information for each block
                - width: Output width in characters
                - height: Output height in characters
        """
        if frame is None or frame.size == 0:
            return {"blocks": [], "colors": [], "edges": [], "width": 0, "height": 0}
        
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        
        h, w = rgb_frame.shape[:2]
        
        # Calculate effective dimensions
        eff_block_w = self.block_width * self.scale_factor
        eff_block_h = self.block_height * self.scale_factor
        
        char_width = w // eff_block_w
        char_height = h // eff_block_h
        
        if target_width:
            char_width = min(char_width, target_width)
        if target_height:
            char_height = min(char_height, target_height)
        
        # Resize frame if needed
        new_w = char_width * eff_block_w
        new_h = char_height * eff_block_h
        
        if new_w != w or new_h != h:
            rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale for edge detection
        gray = rgb_to_gray(rgb_frame)
        
        # Detect edges
        edges = detect_edges(gray, self.algorithm, self.edge_threshold)
        
        # Process blocks in parallel
        result = self._process_blocks_parallel(
            rgb_frame,
            gray,
            edges,
            char_width,
            char_height,
            eff_block_w,
            eff_block_h,
        )
        
        return result
    
    def _process_blocks_parallel(
        self,
        rgb_frame: np.ndarray,
        gray: np.ndarray,
        edges: GradientResult,
        char_width: int,
        char_height: int,
        block_w: int,
        block_h: int,
    ) -> Dict[str, Any]:
        """Process frame blocks using vectorized operations for maximum performance."""
        # Reshape frames for block processing using stride tricks
        # This is much faster than nested loops
        
        h, w = gray.shape[:2]
        
        # Trim to exact block boundaries
        trim_h = char_height * block_h
        trim_w = char_width * block_w
        
        gray_trim = gray[:trim_h, :trim_w]
        
        # Reshape gray to (char_height, block_h, char_width, block_w)
        # Then compute mean over block dimensions
        gray_blocks = gray_trim.reshape(char_height, block_h, char_width, block_w)
        blocks = gray_blocks.mean(axis=(1, 3)) / 255.0
        
        # Process colors if enabled
        colors = None
        if self.color_enabled:
            rgb_trim = rgb_frame[:trim_h, :trim_w]
            rgb_blocks = rgb_trim.reshape(char_height, block_h, char_width, block_w, 3)
            colors = rgb_blocks.mean(axis=(1, 3)).astype(np.uint8)
        
        # Process edges
        mag_trim = edges["magnitude"][:trim_h, :trim_w].astype(np.float32)
        grad_x_trim = edges["gradient_x"][:trim_h, :trim_w]
        grad_y_trim = edges["gradient_y"][:trim_h, :trim_w]
        
        mag_blocks = mag_trim.reshape(char_height, block_h, char_width, block_w)
        gx_blocks = grad_x_trim.reshape(char_height, block_h, char_width, block_w)
        gy_blocks = grad_y_trim.reshape(char_height, block_h, char_width, block_w)
        
        edge_data = np.zeros((char_height, char_width, 3), dtype=np.float32)
        edge_data[:, :, 0] = mag_blocks.mean(axis=(1, 3))
        edge_data[:, :, 1] = gx_blocks.mean(axis=(1, 3))
        edge_data[:, :, 2] = gy_blocks.mean(axis=(1, 3))
        
        return {
            "blocks": blocks.astype(np.float32),
            "colors": colors,
            "edges": edge_data,
            "width": char_width,
            "height": char_height,
        }
    
    def process_batch(
        self,
        frames: List[np.ndarray],
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Process multiple frames in parallel.
        
        Args:
            frames: List of frames to process
            target_width: Target width in characters
            target_height: Target height in characters
            
        Returns:
            List of processed frame data
        """
        futures = [
            self._executor.submit(
                self.process_frame,
                frame,
                target_width,
                target_height,
            )
            for frame in frames
        ]
        
        return [f.result() for f in futures]
    
    def update_quality(self, level: QualityLevel) -> None:
        """Update processing quality level.
        
        Args:
            level: New quality level
        """
        with self._lock:
            if level == QualityLevel.MINIMAL:
                self.scale_factor = 1
                self.block_width = 3
                self.block_height = 6
            elif level == QualityLevel.LOW:
                self.scale_factor = 1
                self.block_width = 2
                self.block_height = 5
            elif level == QualityLevel.STANDARD:
                self.scale_factor = 1
                self.block_width = 2
                self.block_height = 4
            elif level == QualityLevel.HIGH:
                self.scale_factor = 2
                self.block_width = 2
                self.block_height = 4
            elif level == QualityLevel.MAXIMUM:
                self.scale_factor = 4
                self.block_width = 1
                self.block_height = 2
    
    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=False)
    
    def __del__(self):
        """Cleanup resources."""
        try:
            self.shutdown()
        except Exception:
            pass
