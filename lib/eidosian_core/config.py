"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    EIDOSIAN CONFIGURATION SYSTEM                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
Global configuration for the Eidosian decorator system.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Literal
from enum import Enum
class LogLevel(Enum):
    """Log levels for Eidosian logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
@dataclass
class LoggingConfig:
    """Logging configuration."""
    enabled: bool = True
    level: str = "INFO"
    format: str = "[%(asctime)s] %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_args: bool = True
    log_result: bool = True
    log_exceptions: bool = True
    structured: bool = False  # JSON logging
    file_path: Optional[str] = None
    max_arg_length: int = 200
    max_result_length: int = 500
@dataclass
class ProfilingConfig:
    """Profiling configuration."""
    enabled: bool = False
    output_dir: Optional[str] = None
    top_n: int = 20
    sort_by: str = "cumulative"
    include_builtins: bool = False
    save_stats: bool = False
@dataclass  
class BenchmarkingConfig:
    """Benchmarking configuration."""
    enabled: bool = False
    iterations: int = 1
    warmup: int = 0
    output_file: Optional[str] = None
    record_memory: bool = False
    threshold_ms: Optional[float] = None  # Alert if exceeds
@dataclass
class TracingConfig:
    """Tracing configuration."""
    enabled: bool = False
    capture_args: bool = True
    capture_result: bool = True
    capture_locals: bool = False
    max_depth: int = 10
    output_file: Optional[str] = None
@dataclass
class EidosianConfig:
    """
    Master configuration for all Eidosian decorator features.
    
    Can be loaded from environment variables, JSON files, or set programmatically.
    """
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    benchmarking: BenchmarkingConfig = field(default_factory=BenchmarkingConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    
    # Global settings
    auto_revert_on_failure: bool = True
    fail_threshold: int = 3
    dry_run: bool = False
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    def save(self, path: Path) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EidosianConfig":
        """Create from dictionary."""
        return cls(
            logging=LoggingConfig(**data.get("logging", {})),
            profiling=ProfilingConfig(**data.get("profiling", {})),
            benchmarking=BenchmarkingConfig(**data.get("benchmarking", {})),
            tracing=TracingConfig(**data.get("tracing", {})),
            auto_revert_on_failure=data.get("auto_revert_on_failure", True),
            fail_threshold=data.get("fail_threshold", 3),
            dry_run=data.get("dry_run", False),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "EidosianConfig":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def load(cls, path: Path) -> "EidosianConfig":
        """Load configuration from file."""
        path = Path(path)
        if path.exists():
            return cls.from_json(path.read_text())
        return cls()
    
    @classmethod
    def from_env(cls) -> "EidosianConfig":
        """Load configuration from environment variables."""
        config = cls()
        
        # Logging
        if os.getenv("EIDOSIAN_LOG_ENABLED"):
            config.logging.enabled = os.getenv("EIDOSIAN_LOG_ENABLED", "true").lower() == "true"
        if os.getenv("EIDOSIAN_LOG_LEVEL"):
            config.logging.level = os.getenv("EIDOSIAN_LOG_LEVEL", "INFO")
        if os.getenv("EIDOSIAN_LOG_FILE"):
            config.logging.file_path = os.getenv("EIDOSIAN_LOG_FILE")
        if os.getenv("EIDOSIAN_LOG_STRUCTURED"):
            config.logging.structured = os.getenv("EIDOSIAN_LOG_STRUCTURED", "false").lower() == "true"
            
        # Profiling
        if os.getenv("EIDOSIAN_PROFILE_ENABLED"):
            config.profiling.enabled = os.getenv("EIDOSIAN_PROFILE_ENABLED", "false").lower() == "true"
        if os.getenv("EIDOSIAN_PROFILE_OUTPUT"):
            config.profiling.output_dir = os.getenv("EIDOSIAN_PROFILE_OUTPUT")
            
        # Benchmarking  
        if os.getenv("EIDOSIAN_BENCHMARK_ENABLED"):
            config.benchmarking.enabled = os.getenv("EIDOSIAN_BENCHMARK_ENABLED", "false").lower() == "true"
        if os.getenv("EIDOSIAN_BENCHMARK_THRESHOLD"):
            config.benchmarking.threshold_ms = float(os.getenv("EIDOSIAN_BENCHMARK_THRESHOLD", "0"))
            
        # Tracing
        if os.getenv("EIDOSIAN_TRACE_ENABLED"):
            config.tracing.enabled = os.getenv("EIDOSIAN_TRACE_ENABLED", "false").lower() == "true"
            
        return config
# Global configuration instance
_global_config: Optional[EidosianConfig] = None

def get_config() -> EidosianConfig:
    """Get the global Eidosian configuration."""
    global _global_config
    if _global_config is None:
        # Try to load from standard locations
        config_paths = [
            Path.home() / ".eidosian" / "config.json",
            Path.cwd() / ".eidosian.json",
            Path.cwd() / "eidosian_config.json",
        ]
        
        for path in config_paths:
            if path.exists():
                _global_config = EidosianConfig.load(path)
                break
        
        if _global_config is None:
            _global_config = EidosianConfig.from_env()
    
    return _global_config

def set_config(config: EidosianConfig) -> None:
    """Set the global Eidosian configuration."""
    global _global_config
    _global_config = config

def reset_config() -> None:
    """Reset configuration to defaults."""
    global _global_config
    _global_config = None
