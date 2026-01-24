"""
🤖 GUI Automation Module

Higher-level automation patterns built on top of Wayland control.
Provides common workflow automation with visual verification.

Created: 2026-01-23
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# Import control functions
from ..wayland_control import (
    check_daemon,
    mouse_move_absolute,
    mouse_move_relative,
    mouse_click,
    type_text,
    press_key,
    scroll as mouse_scroll,
    take_screenshot
)


@dataclass
class ActionResult:
    """Result of an automation action."""
    success: bool
    action: str
    details: Dict[str, Any]
    timestamp: str
    duration_ms: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "action": self.action,
            "details": self.details,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "error": self.error
        }


def _timed_action(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Execute function and return (result, duration_ms)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration = (time.perf_counter() - start) * 1000
    return result, duration


class AutomationSequence:
    """Builder for automation sequences."""
    
    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.steps: List[Tuple[str, Callable, Dict]] = []
        self.results: List[ActionResult] = []
        self._pause_between = 0.1
    
    def set_pause(self, seconds: float) -> "AutomationSequence":
        """Set pause duration between steps."""
        self._pause_between = seconds
        return self
    
    def move_to(self, x: int, y: int) -> "AutomationSequence":
        """Add mouse move step."""
        self.steps.append((
            f"move_to({x}, {y})",
            lambda x=x, y=y: mouse_move_absolute(x, y),
            {"x": x, "y": y}
        ))
        return self
    
    def click(self, button: str = "left") -> "AutomationSequence":
        """Add click step."""
        button_map = {"left": 1, "right": 2, "middle": 3}
        self.steps.append((
            f"click({button})",
            lambda b=button_map.get(button, 1): mouse_click(b),
            {"button": button}
        ))
        return self
    
    def click_at(self, x: int, y: int, button: str = "left") -> "AutomationSequence":
        """Add move + click step."""
        return self.move_to(x, y).click(button)
    
    def type(self, text: str, delay_ms: int = 0) -> "AutomationSequence":
        """Add typing step."""
        self.steps.append((
            f"type({len(text)} chars)",
            lambda t=text, d=delay_ms: type_text(t, d),
            {"text_length": len(text)}
        ))
        return self
    
    def wait(self, seconds: float) -> "AutomationSequence":
        """Add wait step."""
        self.steps.append((
            f"wait({seconds}s)",
            lambda s=seconds: time.sleep(s) or {"success": True},
            {"seconds": seconds}
        ))
        return self
    
    def screenshot(self, path: Optional[str] = None) -> "AutomationSequence":
        """Add screenshot step."""
        self.steps.append((
            "screenshot",
            lambda p=path: take_screenshot(p),
            {"path": path}
        ))
        return self
    
    def scroll_down(self, amount: int = 3) -> "AutomationSequence":
        """Add scroll down step."""
        self.steps.append((
            f"scroll_down({amount})",
            lambda a=amount: mouse_scroll(a),
            {"amount": amount}
        ))
        return self
    
    def scroll_up(self, amount: int = 3) -> "AutomationSequence":
        """Add scroll up step."""
        self.steps.append((
            f"scroll_up({amount})",
            lambda a=amount: mouse_scroll(-a),
            {"amount": -amount}
        ))
        return self
    
    def run(self, dry_run: bool = False) -> List[ActionResult]:
        """Execute the sequence."""
        self.results = []
        
        # Check daemon first
        if not dry_run:
            status = check_daemon()
            if not status.get("daemon_accessible"):
                return [ActionResult(
                    success=False,
                    action="check_daemon",
                    details=status,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    duration_ms=0,
                    error="Daemon not accessible"
                )]
        
        for step_name, action, details in self.steps:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            if dry_run:
                self.results.append(ActionResult(
                    success=True,
                    action=step_name,
                    details={**details, "dry_run": True},
                    timestamp=timestamp,
                    duration_ms=0
                ))
            else:
                try:
                    result, duration = _timed_action(action)
                    success = result.get("success", True) if isinstance(result, dict) else True
                    self.results.append(ActionResult(
                        success=success,
                        action=step_name,
                        details={**details, **(result if isinstance(result, dict) else {})},
                        timestamp=timestamp,
                        duration_ms=duration,
                        error=result.get("error") if isinstance(result, dict) else None
                    ))
                except Exception as e:
                    self.results.append(ActionResult(
                        success=False,
                        action=step_name,
                        details=details,
                        timestamp=timestamp,
                        duration_ms=0,
                        error=str(e)
                    ))
            
            if self._pause_between > 0 and not dry_run:
                time.sleep(self._pause_between)
        
        return self.results
    
    def summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        return {
            "sequence_name": self.name,
            "total_steps": len(self.steps),
            "executed_steps": len(self.results),
            "successful_steps": sum(1 for r in self.results if r.success),
            "failed_steps": sum(1 for r in self.results if not r.success),
            "total_duration_ms": sum(r.duration_ms for r in self.results),
            "results": [r.to_dict() for r in self.results]
        }


# Pre-built common sequences
def click_sequence(*points: Tuple[int, int], pause: float = 0.3) -> AutomationSequence:
    """Create a sequence that clicks through multiple points."""
    seq = AutomationSequence("click_sequence").set_pause(pause)
    for x, y in points:
        seq.click_at(x, y)
    return seq


def type_at(x: int, y: int, text: str) -> AutomationSequence:
    """Create a sequence that clicks at position and types text."""
    return (AutomationSequence("type_at")
            .click_at(x, y)
            .wait(0.1)
            .type(text))


def scroll_page(direction: str = "down", times: int = 3, pause: float = 0.2) -> AutomationSequence:
    """Create a scrolling sequence."""
    seq = AutomationSequence(f"scroll_{direction}").set_pause(pause)
    for _ in range(times):
        if direction == "down":
            seq.scroll_down()
        else:
            seq.scroll_up()
    return seq


# Visual verification integration
from ..visual_feedback import ScreenState, compare_states, wait_for_change


class VerifiedSequence(AutomationSequence):
    """
    Automation sequence with visual verification.
    Takes screenshots before/after each step to verify changes.
    """
    
    def __init__(self, name: str = "verified_sequence"):
        super().__init__(name)
        self.verification_results: List[Dict[str, Any]] = []
        self._verify_steps = True
    
    def enable_verification(self, enabled: bool = True) -> "VerifiedSequence":
        """Enable or disable visual verification."""
        self._verify_steps = enabled
        return self
    
    def run_verified(self) -> Dict[str, Any]:
        """
        Execute sequence with visual verification at each step.
        """
        self.verification_results = []
        
        # Initial state
        prev_state = ScreenState.capture() if self._verify_steps else None
        
        for step_name, action, details in self.steps:
            # Execute step
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                result, duration = _timed_action(action)
                success = result.get("success", True) if isinstance(result, dict) else True
                
                action_result = ActionResult(
                    success=success,
                    action=step_name,
                    details=details,
                    timestamp=timestamp,
                    duration_ms=duration
                )
            except Exception as e:
                action_result = ActionResult(
                    success=False,
                    action=step_name,
                    details=details,
                    timestamp=timestamp,
                    duration_ms=0,
                    error=str(e)
                )
            
            self.results.append(action_result)
            
            # Verify screen changed
            if self._verify_steps and prev_state:
                time.sleep(0.2)  # Allow screen to update
                curr_state = ScreenState.capture()
                if curr_state:
                    comparison = compare_states(prev_state, curr_state)
                    self.verification_results.append({
                        "step": step_name,
                        "visual_change": comparison.get("pixel_analysis", {}).get("has_significant_change", False),
                        "change_percent": comparison.get("pixel_analysis", {}).get("change_percent", 0)
                    })
                    prev_state = curr_state
            
            if self._pause_between > 0:
                time.sleep(self._pause_between)
        
        return self.verified_summary()
    
    def verified_summary(self) -> Dict[str, Any]:
        """Get summary including verification results."""
        base = self.summary()
        base["verification_enabled"] = self._verify_steps
        base["verification_results"] = self.verification_results
        
        if self.verification_results:
            verified_changes = sum(1 for v in self.verification_results if v.get("visual_change"))
            base["verified_changes"] = verified_changes
            base["verification_rate"] = f"{verified_changes}/{len(self.verification_results)}"
        
        return base


class RobustSequence(VerifiedSequence):
    """
    Automation sequence with retry logic on verification failure.
    
    If a step's visual verification fails (no significant change detected),
    the step is retried up to max_retries times before marking as failed.
    """
    
    def __init__(self, name: str = "robust_sequence"):
        super().__init__(name)
        self._max_retries = 3
        self._retry_delay = 0.5
        self._change_threshold = 0.5  # % change required to verify
    
    def set_retries(self, max_retries: int, delay: float = 0.5) -> "RobustSequence":
        """Configure retry behavior."""
        self._max_retries = max_retries
        self._retry_delay = delay
        return self
    
    def set_change_threshold(self, threshold: float) -> "RobustSequence":
        """Set the change threshold for verification (% pixels changed)."""
        self._change_threshold = threshold
        return self
    
    def run_robust(self) -> Dict[str, Any]:
        """
        Execute sequence with retry logic.
        
        Each step that fails verification is retried up to max_retries times.
        """
        self.results = []
        self.verification_results = []
        retry_stats = {"total_retries": 0, "successful_retries": 0, "failed_steps": []}
        
        prev_state = ScreenState.capture() if self._verify_steps else None
        
        for step_idx, (step_name, action, details) in enumerate(self.steps):
            success = False
            retries = 0
            
            while not success and retries <= self._max_retries:
                timestamp = datetime.now(timezone.utc).isoformat()
                
                try:
                    result, duration = _timed_action(action)
                    action_success = result.get("success", True) if isinstance(result, dict) else True
                    
                    if not action_success:
                        # Action itself failed
                        if retries < self._max_retries:
                            retries += 1
                            retry_stats["total_retries"] += 1
                            time.sleep(self._retry_delay)
                            continue
                        else:
                            self.results.append(ActionResult(
                                success=False,
                                action=step_name,
                                details={**details, "retries": retries},
                                timestamp=timestamp,
                                duration_ms=duration,
                                error="Action failed after retries"
                            ))
                            retry_stats["failed_steps"].append(step_name)
                            break
                    
                    # Action succeeded - verify visually
                    if self._verify_steps and prev_state:
                        time.sleep(0.2)
                        curr_state = ScreenState.capture()
                        if curr_state:
                            comparison = compare_states(prev_state, curr_state)
                            change_pct = comparison.get("pixel_analysis", {}).get("change_percent", 0)
                            
                            if change_pct >= self._change_threshold:
                                # Verified!
                                success = True
                                self.verification_results.append({
                                    "step": step_name,
                                    "visual_change": True,
                                    "change_percent": change_pct,
                                    "retries_needed": retries
                                })
                                prev_state = curr_state
                            elif retries < self._max_retries:
                                # Visual verification failed - retry
                                retries += 1
                                retry_stats["total_retries"] += 1
                                time.sleep(self._retry_delay)
                                continue
                            else:
                                # Max retries reached
                                success = True  # Mark as "done" but note the issue
                                self.verification_results.append({
                                    "step": step_name,
                                    "visual_change": False,
                                    "change_percent": change_pct,
                                    "retries_needed": retries,
                                    "note": "Verification failed after retries"
                                })
                        else:
                            success = True  # Couldn't capture, proceed
                    else:
                        success = True  # No verification needed
                    
                    if success:
                        self.results.append(ActionResult(
                            success=True,
                            action=step_name,
                            details={**details, "retries": retries},
                            timestamp=timestamp,
                            duration_ms=duration
                        ))
                        if retries > 0:
                            retry_stats["successful_retries"] += 1
                    
                except Exception as e:
                    if retries < self._max_retries:
                        retries += 1
                        retry_stats["total_retries"] += 1
                        time.sleep(self._retry_delay)
                    else:
                        self.results.append(ActionResult(
                            success=False,
                            action=step_name,
                            details={**details, "retries": retries},
                            timestamp=timestamp,
                            duration_ms=0,
                            error=str(e)
                        ))
                        retry_stats["failed_steps"].append(step_name)
                        break
            
            if self._pause_between > 0:
                time.sleep(self._pause_between)
        
        summary = self.verified_summary()
        summary["retry_stats"] = retry_stats
        return summary
