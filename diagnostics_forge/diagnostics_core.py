import logging
import json
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from logging.handlers import RotatingFileHandler

class DiagnosticsForge:
    """
    Handles structured logging, performance metrics, and log rotation.
    """
    
    def __init__(self, log_dir: str = "logs", service_name: str = "eidos", max_bytes: int = 10*1024*1024, backup_count: int = 5):
        self.service_name = service_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"{service_name}.log"
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        self._setup_logging()
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}

    def _setup_logging(self):
        self.logger = logging.getLogger(self.service_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers if logger already initialized
        if not self.logger.handlers:
            # Rotating File Handler
            fh = RotatingFileHandler(
                self.log_file, 
                maxBytes=self.max_bytes, 
                backupCount=self.backup_count
            )
            fh.setLevel(logging.DEBUG)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            
            # Standard Eidosian Format
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def log_event(self, level: str, message: str, **kwargs):
        """Log a structured event with optional context data."""
        extra = kwargs
        # Use a more machine-parsable separator for structured data in text logs
        msg = f"{message} | DATA: {json.dumps(extra)}" if extra else message
        
        lvl = level.upper()
        if lvl == "INFO":
            self.logger.info(msg)
        elif lvl == "DEBUG":
            self.logger.debug(msg)
        elif lvl == "ERROR":
            self.logger.error(msg, exc_info=kwargs.get('exc_info', False))
        elif lvl == "WARNING":
            self.logger.warning(msg)
        elif lvl == "CRITICAL":
            self.logger.critical(msg)

    def start_timer(self, name: str) -> Dict[str, Any]:
        """Start a performance timer."""
        return {"name": name, "start": time.perf_counter()}

    def stop_timer(self, timer: Dict[str, Any]) -> float:
        """Stop a performance timer and record the duration."""
        duration = time.perf_counter() - timer["start"]
        name = timer["name"]
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "duration_sec": duration
        }
        self.metrics[name].append(metric_entry)
        self.log_event("DEBUG", f"Metric recorded: {name}", duration=duration)
        return duration

    def get_metrics_summary(self, name: str) -> Dict[str, Any]:
        """Calculate aggregate statistics for a given metric."""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        durations = [m["duration_sec"] for m in self.metrics[name]]
        return {
            "count": len(durations),
            "avg": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "total": sum(durations)
        }

    def save_metrics(self, path: Optional[Union[str, Path]] = None):
        """Export all recorded metrics to a JSON file."""
        if path is None:
            path = self.log_dir / f"{self.service_name}_metrics.json"
        
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def clear_metrics(self):
        """Reset internal metrics storage."""
        self.metrics = {}
