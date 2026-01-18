#!/usr/bin/env python3
"""
🔹 Eidos Configuration Module 🔹

Centralizes all project-wide settings (model names, thresholds, API endpoints, etc.).
Supports environment-variable overrides to avoid hard-coded changes in code.

Features:
- Pydantic models for type validation and better IDE support
- Automatic scanning of project files for potential config variables
- Backward compatibility with existing global variables
- Fallback mechanisms for missing dependencies
"""

import os
import re
import sys
import logging
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union, TypeVar, cast

# Try to import Pydantic, but provide fallbacks if not available
try:
    from pydantic import BaseSettings, Field, validator
    from pydantic.env_settings import SettingsSourceCallable
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Create mock classes/functions for backward compatibility
    PYDANTIC_AVAILABLE = False
    
    class BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def Field(*args, **kwargs):
        return None
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    SettingsSourceCallable = Any

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
def get_env_value(name: str, default: Any) -> Any:
    """Get environment variable with type conversion based on default value."""
    value = os.getenv(name)
    if value is None:
        return default
    
    # Try to convert to the same type as the default
    if isinstance(default, bool):
        return value.lower() in ("true", "1", "yes", "y", "t")
    elif isinstance(default, int):
        return int(value)
    elif isinstance(default, float):
        return float(value)
    else:
        return value

# -----------------------------------------------------------------------------
# LOG LEVEL
# -----------------------------------------------------------------------------
LOG_LEVEL_NAME = os.getenv("EIDOS_LOG_LEVEL", "DEBUG")
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME.upper(), logging.DEBUG)

# -----------------------------------------------------------------------------
# LLM & RETRIEVER SETTINGS
# -----------------------------------------------------------------------------
DEFAULT_OLLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
DEFAULT_SENTENCE_TRANSFORMER_MODEL = os.getenv("EIDOS_ST_MODEL", "all-MiniLM-L6-v2")
DEFAULT_LLM_BACKEND = os.getenv("EIDOS_LLM_BACKEND", "ollama")
DEFAULT_LLM_MODEL = os.getenv("EIDOS_LLM_MODEL", "deepseek-r1:1.5b")
DEFAULT_EVOLUTION_THRESHOLD = int(os.getenv("EIDOS_EVOLUTION_THRESHOLD", "3"))
DEFAULT_TEMPERATURE = float(os.getenv("EIDOS_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("EIDOS_MAX_TOKENS", "8196"))

# -----------------------------------------------------------------------------
# RETRY LOGIC
# -----------------------------------------------------------------------------
MAX_RETRIES = int(os.getenv("EIDOS_MAX_RETRIES", "3"))
BACKOFF_SECONDS = float(os.getenv("EIDOS_BACKOFF_SECONDS", "1.0"))

# -----------------------------------------------------------------------------
# PYDANTIC MODELS (if available)
# -----------------------------------------------------------------------------
if PYDANTIC_AVAILABLE:
    class LoggingSettings(BaseSettings):
        """Logging configuration settings."""
        level: str = Field(default="DEBUG", env="EIDOS_LOG_LEVEL")
        
        @property
        def log_level(self) -> int:
            """Convert string log level to logging module constant."""
            return getattr(logging, self.level.upper(), logging.DEBUG)
        
        class Config:
            env_prefix = "EIDOS_"
    
    class LLMSettings(BaseSettings):
        """LLM and retriever configuration settings."""
        api_base: str = Field(default="http://localhost:11434", env="OLLAMA_API_BASE")
        model: str = Field(default="deepseek-r1:1.5b", env="EIDOS_LLM_MODEL")
        backend: str = Field(default="ollama", env="EIDOS_LLM_BACKEND")
        sentence_transformer_model: str = Field(default="all-MiniLM-L6-v2", env="EIDOS_ST_MODEL")
        evolution_threshold: int = Field(default=3, env="EIDOS_EVOLUTION_THRESHOLD")
        temperature: float = Field(default=0.7, env="EIDOS_TEMPERATURE")
        max_tokens: int = Field(default=8196, env="EIDOS_MAX_TOKENS")
        
        class Config:
            env_prefix = "EIDOS_"
    
    class RetrySettings(BaseSettings):
        """Retry configuration settings."""
        max_retries: int = Field(default=3, env="EIDOS_MAX_RETRIES")
        backoff_seconds: float = Field(default=1.0, env="EIDOS_BACKOFF_SECONDS")
        
        class Config:
            env_prefix = "EIDOS_"
    
    class EidosConfig(BaseSettings):
        """Main configuration class that combines all settings."""
        logging: LoggingSettings = Field(default_factory=LoggingSettings)
        llm: LLMSettings = Field(default_factory=LLMSettings)
        retry: RetrySettings = Field(default_factory=RetrySettings)
        
        class Config:
            env_prefix = "EIDOS_"
            
        def update_from_env(self) -> None:
            """Refresh settings from environment variables."""
            self.logging = LoggingSettings()
            self.llm = LLMSettings()
            self.retry = RetrySettings()
else:
    @dataclass
    class LoggingSettings:
        """Logging configuration settings."""
        level: str = field(default="DEBUG")
        
        @property
        def log_level(self) -> int:
            """Convert string log level to logging module constant."""
            return getattr(logging, self.level.upper(), logging.DEBUG)
    
    @dataclass
    class LLMSettings:
        """LLM and retriever configuration settings."""
        api_base: str = field(default="http://localhost:11434")
        model: str = field(default="deepseek-r1:1.5b")
        backend: str = field(default="ollama")
        sentence_transformer_model: str = field(default="all-MiniLM-L6-v2")
        evolution_threshold: int = field(default=3)
        temperature: float = field(default=0.7)
        max_tokens: int = field(default=8196)
    
    @dataclass
    class RetrySettings:
        """Retry configuration settings."""
        max_retries: int = field(default=3)
        backoff_seconds: float = field(default=1.0)
    
    @dataclass
    class EidosConfig:
        """Main configuration class that combines all settings."""
        logging: LoggingSettings = field(default_factory=LoggingSettings)
        llm: LLMSettings = field(default_factory=LLMSettings)
        retry: RetrySettings = field(default_factory=RetrySettings)
        
        def update_from_env(self) -> None:
            """Refresh settings from environment variables."""
            self.logging.level = os.getenv("EIDOS_LOG_LEVEL", "DEBUG")
            
            self.llm.api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            self.llm.model = os.getenv("EIDOS_LLM_MODEL", "deepseek-r1:1.5b")
            self.llm.backend = os.getenv("EIDOS_LLM_BACKEND", "ollama")
            self.llm.sentence_transformer_model = os.getenv("EIDOS_ST_MODEL", "all-MiniLM-L6-v2")
            self.llm.evolution_threshold = int(os.getenv("EIDOS_EVOLUTION_THRESHOLD", "3"))
            self.llm.temperature = float(os.getenv("EIDOS_TEMPERATURE", "0.7"))
            self.llm.max_tokens = int(os.getenv("EIDOS_MAX_TOKENS", "8196"))
            
            self.retry.max_retries = int(os.getenv("EIDOS_MAX_RETRIES", "3"))
            self.retry.backoff_seconds = float(os.getenv("EIDOS_BACKOFF_SECONDS", "1.0"))

# Create a global config instance
config = EidosConfig()

# -----------------------------------------------------------------------------
# CONFIG FILE SCANNER
# -----------------------------------------------------------------------------
def scan_for_config_vars(directory: Union[str, Path] = None) -> Dict[str, Any]:
    """
    Scan Python files in the specified directory for potential config variables.
    
    Args:
        directory: Directory to scan. Defaults to the directory of this file.
        
    Returns:
        Dictionary of discovered configuration variables and their values.
    """
    if directory is None:
        directory = Path(__file__).parent
    
    directory = Path(directory)
    if not directory.exists():
        return {}
    
    # Pattern to match likely config variables (uppercase with underscores)
    pattern = re.compile(r'^([A-Z][A-Z0-9_]*)\s*=\s*(.+)$', re.MULTILINE)
    
    # Dictionary to store discovered variables
    discovered_vars = {}  # type: Dict[str, Any]
    existing_vars = set(globals().keys())
    
    # Scan all Python files in the directory
    for py_file in directory.glob('**/*.py'):
        # Skip this file to avoid recursion
        if py_file.name == Path(__file__).name:
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find all potential config variables
            matches = pattern.findall(content)
            for var_name, var_value in matches:
                # Skip if already in our config
                if var_name in existing_vars or var_name.startswith('_'):
                    continue
                
                # Try to evaluate the value safely
                try:
                    # Only allow built-in types
                    safe_globals = {
                        'True': True, 'False': False, 'None': None,
                        'int': int, 'float': float, 'str': str, 'bool': bool,
                        'list': list, 'dict': dict, 'tuple': tuple, 'set': set
                    }
                    value = eval(var_value, safe_globals, {})
                    discovered_vars[var_name] = value
                except (SyntaxError, NameError, TypeError):
                    # If we can't evaluate, store as string
                    discovered_vars[var_name] = var_value.strip()
        
        except Exception as e:
            logging.warning(f"Error scanning file {py_file}: {e}")
    
    return discovered_vars

def apply_discovered_configs(discovered_vars: Dict[str, Any]) -> None:
    """
    Apply discovered configuration variables to the global namespace.
    
    Args:
        discovered_vars: Dictionary of variable names and values to apply.
    """
    for var_name, var_value in discovered_vars.items():
        globals()[var_name] = var_value
        
    return len(discovered_vars)

def scan_and_update_config(directory: Union[str, Path] = None) -> int:
    """
    Scan for config variables and update the global namespace.
    
    Args:
        directory: Directory to scan. Defaults to the directory of this file.
        
    Returns:
        Number of variables added to the configuration.
    """
    discovered = scan_for_config_vars(directory)
    return apply_discovered_configs(discovered)

# -----------------------------------------------------------------------------
# On-import actions: Update config from env and provide easy access attributes
# -----------------------------------------------------------------------------
# Update settings from environment variables
config.update_from_env()

# For backward compatibility, ensure the original variables have the updated values
# This allows existing code to continue using the global variables
LOG_LEVEL_NAME = config.logging.level
LOG_LEVEL = config.logging.log_level
DEFAULT_OLLLAMA_API_BASE = config.llm.api_base
DEFAULT_LLM_MODEL = config.llm.model
DEFAULT_LLM_BACKEND = config.llm.backend
DEFAULT_SENTENCE_TRANSFORMER_MODEL = config.llm.sentence_transformer_model
DEFAULT_EVOLUTION_THRESHOLD = config.llm.evolution_threshold
DEFAULT_TEMPERATURE = config.llm.temperature
MAX_TOKENS = config.llm.max_tokens
MAX_RETRIES = config.retry.max_retries
BACKOFF_SECONDS = config.retry.backoff_seconds

# If this module is run directly, perform a config scan
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--scan":
        print("Scanning for configuration variables...")
        directory = sys.argv[2] if len(sys.argv) > 2 else None
        count = scan_and_update_config(directory)
        print(f"Found and added {count} new configuration variables.")
        
        # Print current configuration
        print("\nCurrent configuration:")
        for var_name, var_value in sorted(globals().items()):
            if var_name.isupper() and not var_name.startswith('_'):
                print(f"{var_name} = {var_value}")
    else:
        print("EidosConfig - Configuration Management")
        print("Usage:")
        print("  python -m eidos_config --scan [directory]  # Scan for config variables")
