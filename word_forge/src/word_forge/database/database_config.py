"""

This module provides a comprehensive configuration system for database connections,
performance tuning, and query management in Word Forge. It centralizes all database-related
configuration parameters in a type-safe, immutable, and extensible structure.

Key Features:
- Type-safe database configuration with dataclasses
- Connection pool management with different pooling strategies
- SQLite performance optimization through pragmas
- Transaction isolation level configuration
- Read and write optimized configuration profiles
- Validation and error handling with result types
- Environment variable mapping for configuration overrides

Classes:
    DatabaseConfig: Central class for database configuration management
    PoolSettingsDict: Type definition for connection pool settings
    PragmaDict: Type alias for database pragma dictionaries

    # Basic usage with default configuration

    config = DatabaseConfig()
    db_path = config.get_db_path
    connection_uri = config.get_connection_uri()

    # Creating optimized configurations
    read_config = config.optimize_for_reads()
    write_config = config.optimize_for_writes()

    # Custom configuration with specific settings
    custom_config = DatabaseConfig(
        db_path="/path/to/custom.db",
        enable_wal_mode=True,
        cache_size=5000

    The module is structured around the central DatabaseConfig class, which manages
    different aspects of database configuration through a combination of attributes,
    cached properties, and methods. It follows an immutable pattern where configuration
    changes create new instances rather than modifying existing ones.

Dependencies:
    - word_forge.configs.config_essentials: Core configuration types and utilities
    - Standard library: dataclasses, functools, pathlib, typing

Notes:
    - Currently supports SQLite as the primary database dialect
    - Configuration validation ensures consistency and correctness of settings
    - Performance optimization profiles for different workload types
Database configuration system for Word Forge.

This module defines the configuration schema for database connections,
performance parameters, SQL templates, and connection pooling used
throughout the Word Forge system.

Architecture:
    ┌─────────────────────┐
    │   DatabaseConfig    │
    └───────────┬─────────┘
                │
    ┌───────────┴─────────┐
    │     Components      │
    └─────────────────────┘
    ┌─────┬─────┬─────┬───────┬─────┐
    │Conn │Pool │SQL  │Pragmas│Trans│
    └─────┴─────┴─────┴───────┴─────┘
"""

from dataclasses import dataclass, field, fields, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Literal, Set, TypedDict, Union, cast

from word_forge.configs.config_essentials import (
    DATA_ROOT,
    ConnectionPoolMode,
    DatabaseConfigError,
    DatabaseDialect,
    ErrorCategory,
    ErrorSeverity,
    PathLike,
    Result,
    SQLitePragmas,
    SQLTemplates,
    TransactionIsolationLevel,
    measure_execution,
)


# More precise type definitions
class PoolSettingsDict(TypedDict):
    """Type definition for connection pool settings."""

    mode: ConnectionPoolMode
    size: int
    timeout: float
    recycle: int


# Type aliases for clearer return signatures
PragmaDict = Dict[str, str]


# Define type-safe attributes for database configuration
DatabaseConfigAttrs = Literal[
    "db_path",
    "dialect",
    "pragmas",
    "sql_templates",
    "pool_mode",
    "pool_size",
    "pool_timeout",
    "pool_recycle",
    "isolation_level",
    "enable_foreign_keys",
    "enable_wal_mode",
    "page_size",
    "cache_size",
]


@dataclass
class DatabaseConfig:
    """
        Database configuration for SQL database connections.

        Controls SQLite database location, connection parameters, SQL query templates,
        performance optimizations, and connection pooling settings.

        Attributes:
            db_path: Path to the SQLite database file
            dialect: Database dialect (currently supports SQLite)
            pragmas: SQLite pragmas for performance optimization
            sql_templates: SQL query templates for database operations
            pool_mode: Connection pooling mode (fixed, dynamic, none)
            pool_size: Connection pool size (for fixed/dynamic modes)
            pool_timeout: Connection pool timeout in seconds
            pool_recycle: Time in seconds between connection recycling
            isolation_level: Transaction isolation level
            enable_foreign_keys: Whether to enforce foreign key constraints
            enable_wal_mode: Whether to use WAL journal mode (SQLite only)
            page_size: Database page size in bytes (SQLite only)
            cache_size: Memory pages to use for database cache (SQLite only)

        Usage:
            ```python
            from word_forge.config import config
    from word_forge.database.database_config import DatabaseConfig

            # Get database path
            db_path = config.database.get_db_path

            # Get connection URI with pragmas
            uri = config.database.get_connection_uri()

            # Create a new configuration with a different path
            test_config = config.database.with_path("path/to/test.db")
            ```
    """

    # Core database settings
    db_path: str = str(DATA_ROOT / "word_forge.sqlite")
    dialect: DatabaseDialect = DatabaseDialect.SQLITE

    # SQLite pragmas for performance optimization
    pragmas: SQLitePragmas = field(
        default_factory=lambda: {
            "foreign_keys": "ON",
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "cache_size": "-2000",  # 2MB cache
            "temp_store": "MEMORY",
        }
    )

    # SQL query templates for database operations
    sql_templates: SQLTemplates = field(
        default_factory=lambda: cast(
            SQLTemplates,
            {
                "create_words_table": """
                CREATE TABLE IF NOT EXISTS words (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT UNIQUE NOT NULL,
                    definition TEXT,
                    part_of_speech TEXT,
                    usage_examples TEXT,
                    last_refreshed REAL
                );
            """,
                "create_relationships_table": """
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word_id INTEGER NOT NULL,
                    related_term TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    FOREIGN KEY(word_id) REFERENCES words(id)
                );
            """,
                "create_word_id_index": """
                CREATE INDEX IF NOT EXISTS idx_relationships_word_id
                ON relationships(word_id);
            """,
                "create_unique_relationship_index": """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_relationship
                ON relationships(word_id, related_term, relationship_type);
            """,
            },
        )
    )

    # Connection pool settings
    pool_mode: ConnectionPoolMode = "fixed"
    pool_size: int = 5
    pool_timeout: float = 30.0
    pool_recycle: int = 3600  # Recycle connections after 1 hour

    # Transaction settings
    isolation_level: TransactionIsolationLevel = "SERIALIZABLE"

    # Performance settings
    enable_foreign_keys: bool = True
    enable_wal_mode: bool = True
    page_size: int = 4096  # 4KB pages
    cache_size: int = 2000  # 2000 pages (~8MB)

    # ==========================================
    # Cached Properties
    # ==========================================

    @cached_property
    def db_path_object(self) -> Path:
        """
        Database path as a Path object.

        Returns:
            Path: Object representing the database location.

        Example:
            ```python
            path_obj = config.database.db_path_object
            parent_dir = path_obj.parent
            ```
        """
        return Path(self.db_path)

    @cached_property
    def get_db_path(self) -> Path:
        """
        Get database path as a validated Path object.

        Ensures the parent directory exists for SQLite databases.

        Returns:
            Path: Database file location.

        Raises:
            DatabaseConfigError: If path is invalid or parent directory does not exist

        Example:
            ```python
            db_path = config.database.get_db_path
            print(f"Using database at {db_path}")
            ```
        """
        path = self.db_path_object

        # For SQLite, ensure parent directory exists
        if self.dialect == DatabaseDialect.SQLITE:
            parent_dir = path.parent
            if not parent_dir.exists():
                raise DatabaseConfigError(
                    f"Database parent directory does not exist: {parent_dir}"
                )

        return path

    @cached_property
    def effective_pragmas(self) -> PragmaDict:
        """
        Get the effective pragmas with computed values.

        Ensures all explicitly set performance parameters are
        reflected in the pragmas dictionary.

        Returns:
            Dict[str, str]: Effective pragmas dictionary with all settings applied

        Example:
            ```python
            pragma_dict = config.database.effective_pragmas
            print(f"Using cache size of {pragma_dict['cache_size']}")
            ```
        """
        # Start with user-defined pragmas
        result: Dict[str, str] = {}

        # Copy with proper type conversion
        for k, v in self.pragmas.items():
            result[k] = str(v)

        # Apply explicit settings if not present in pragmas
        if self.enable_foreign_keys and "foreign_keys" not in result:
            result["foreign_keys"] = "ON"

        if self.enable_wal_mode and "journal_mode" not in result:
            result["journal_mode"] = "WAL"

        # Apply cache size setting
        if "cache_size" not in result:
            result["cache_size"] = str(-self.cache_size)  # Negative means KB

        # Apply page size setting
        if "page_size" not in result:
            result["page_size"] = str(self.page_size)

        return result

    # ==========================================
    # Public Methods
    # ==========================================

    def get_connection_uri(self) -> str:
        """
        Build SQLite connection URI with pragmas.

        Creates a URI that includes all effective pragmas as query parameters.

        Returns:
            str: SQLite connection string with pragmas as query parameters.

        Raises:
            DatabaseConfigError: If dialect is not SQLite

        Example:
            ```python
            uri = config.database.get_connection_uri()
            conn = sqlite3.connect(uri, uri=True)
            ```
        """
        if self.dialect != DatabaseDialect.SQLITE:
            raise DatabaseConfigError(
                f"Connection URI for {self.dialect.name} is not implemented"
            )

        params = "&".join(f"{k}={v}" for k, v in self.effective_pragmas.items())
        return f"file:{self.db_path}?{params}"

    def get_pool_settings(self) -> PoolSettingsDict:
        """
        Get connection pool settings as a dictionary.

        Returns:
            PoolSettingsDict: Pool configuration for database engines

        Example:
            ```python
            pool_config = config.database.get_pool_settings()
            engine = create_engine(url, **pool_config)
            ```
        """
        return {
            "mode": self.pool_mode,
            "size": self.pool_size if self.pool_mode != "none" else 0,
            "timeout": self.pool_timeout,
            "recycle": self.pool_recycle,
        }

    def with_path(self, path: PathLike) -> "DatabaseConfig":
        """
        Create a new instance with a modified path.

        Args:
            path: New database path (string or Path object)

        Returns:
            DatabaseConfig: New instance with updated path

        Example:
            ```python
            test_config = config.database.with_path("tests/test_db.sqlite")
            ```
        """
        return self._create_modified_instance(db_path=str(path))

    def with_dialect(self, dialect: DatabaseDialect) -> "DatabaseConfig":
        """
        Create a new instance with a different database dialect.

        Args:
            dialect: New database dialect

        Returns:
            DatabaseConfig: New instance with updated dialect

        Example:
            ```python
            memory_config = config.database.with_dialect(DatabaseDialect.MEMORY)
            ```
        """
        return self._create_modified_instance(dialect=dialect)

    def optimize_for_reads(self) -> "DatabaseConfig":
        """
        Create a new instance optimized for read operations.

        Configures the database for optimal read performance with:
        - Larger connection pool
        - Larger cache size
        - Memory-based temporary storage
        - WAL journal mode
        - Relaxed transaction isolation

        Returns:
            DatabaseConfig: New read-optimized configuration

        Example:
            ```python
            read_config = config.database.optimize_for_reads()
            read_conn = create_connection(read_config)
            ```
        """
        with measure_execution("database.optimize_for_reads"):
            read_pragmas: Dict[str, str] = {}

            # Copy existing pragmas
            for k, v in self.pragmas.items():
                read_pragmas[k] = str(v)

            # Update with read-optimized values
            read_pragmas.update(
                {
                    "synchronous": "NORMAL",  # Less durability, more speed
                    "journal_mode": "WAL",  # Allow concurrent reads/writes
                    "cache_size": str(-8000),  # Larger cache for read-heavy workloads
                    "temp_store": "MEMORY",  # Keep temp tables in memory
                }
            )

            return self._create_modified_instance(
                pragmas=cast(Dict[str, Any], read_pragmas),
                pool_size=max(
                    self.pool_size, 8
                ),  # Larger pool for more concurrent reads
                isolation_level="READ_COMMITTED",  # Lower isolation level for better concurrency
                enable_wal_mode=True,  # WAL is critical for read performance
                cache_size=8000,  # Larger cache size
            )

    def optimize_for_writes(self) -> "DatabaseConfig":
        """
        Create a new instance optimized for write operations.

        Configures the database for optimal write performance with:
        - Moderate connection pool
        - Memory mapping for large datasets
        - Adjusted synchronous mode
        - WAL journal mode
        - Larger page size

        Returns:
            DatabaseConfig: New write-optimized configuration

        Example:
            ```python
            write_config = config.database.optimize_for_writes()
            write_conn = create_connection(write_config)
            ```
        """
        with measure_execution("database.optimize_for_writes"):
            write_pragmas: Dict[str, str] = {}

            # Copy existing pragmas
            for k, v in self.pragmas.items():
                write_pragmas[k] = str(v)

            # Update with write-optimized values
            write_pragmas.update(
                {
                    "synchronous": "NORMAL",  # Balance between durability and performance
                    "journal_mode": "WAL",  # Better for writes than DELETE
                    "cache_size": str(-4000),  # Moderate cache
                    # Using a known pragma instead of mmap_size which isn't in SQLitePragmas
                    "temp_store": "MEMORY",  # Store temp data in memory
                }
            )

            return self._create_modified_instance(
                pragmas=cast(Dict[str, Any], write_pragmas),
                pool_size=max(self.pool_size, 4),  # Moderate pool size for writes
                pool_recycle=1800,  # More frequent recycling for write connections
                enable_wal_mode=True,  # WAL is also good for writes
                page_size=8192,  # Larger page size for bulk writes
                cache_size=4000,  # Moderate cache size
            )

    def validate(self) -> Result[None]:
        """
        Validate the entire configuration for consistency and correctness.

        Checks all configuration parameters for valid values and
        compatibility with the selected database dialect.

        Returns:
            Result[None]: Success if validation passes or error details if it fails

        Example:
            ```python
            result = config.database.validate()
            if result.is_success:
                print("Configuration is valid")
            else:
                print(f"Invalid configuration: {result.error}")
            ```
        """
        with measure_execution("database.validate"):
            errors: list[str] = []

            # Validate dialect-specific settings
            if self.dialect == DatabaseDialect.SQLITE:
                # Verify db_path is set for SQLite
                if not self.db_path:
                    errors.append("SQLite database path cannot be empty")

                # Validate pool settings
                if self.pool_size <= 0 and self.pool_mode != "none":
                    errors.append(
                        f"Pool size must be positive when mode is {self.pool_mode}, got {self.pool_size}"
                    )

                # Validate timeout
                if self.pool_timeout <= 0:
                    errors.append(
                        f"Pool timeout must be positive, got {self.pool_timeout}"
                    )

            if errors:
                # Convert list to string representation for context
                errors_str = "; ".join(errors)
                return Result[None].failure(
                    code="CONFIG_VALIDATION_ERROR",
                    message=f"Configuration validation failed: {errors_str}",
                    context={
                        "errors": errors_str,
                        "database_config": self.db_path,
                    },
                    category=ErrorCategory.VALIDATION,
                    severity=ErrorSeverity.ERROR,
                )

            return Result[None].success(None)

    def create_directory_if_needed(self) -> Result[None]:
        """
        Ensure the database directory exists.

        Creates all parent directories for the database file path if they don't exist.

        Returns:
            Result[None]: Success if directory exists or was created, or error details if it fails

        Example:
            ```python
            result = config.database.create_directory_if_needed()
            if not result.is_success:
                print(f"Failed to create directory: {result.error}")
            ```
        """
        with measure_execution("database.create_directory") as metrics:
            try:
                path = self.db_path_object
                parent_dir = path.parent

                if not parent_dir.exists():
                    parent_dir.mkdir(parents=True, exist_ok=True)
                    metrics.context["created_directory"] = True
                else:
                    metrics.context["created_directory"] = False

                return Result[None].success(None)
            except Exception as e:
                return Result[None].failure(
                    code="DIRECTORY_CREATE_ERROR",
                    message=f"Failed to create database directory: {str(e)}",
                    context={
                        "path": str(self.db_path_object),
                        "error": str(e),
                    },
                    category=ErrorCategory.RESOURCE,
                    severity=ErrorSeverity.ERROR,
                )

    # ==========================================
    # Private Helper Methods
    # ==========================================

    def _create_modified_instance(
        self,
        **kwargs: Union[
            str,
            int,
            float,
            bool,
            Dict[str, Any],
            DatabaseDialect,
            ConnectionPoolMode,
            TransactionIsolationLevel,
        ],
    ) -> "DatabaseConfig":
        """
        Create a new configuration instance with modified attributes.

        This method ensures immutability by creating a new instance rather than
        modifying the current one, preserving the original configuration intact.
        All type checking is enforced by the replace() function.

        Args:
            **kwargs: Attribute name-value pairs to override. Valid keys are the attribute
                      names of the DatabaseConfig class.

        Returns:
            DatabaseConfig: New instance with specified modifications while preserving
                           all other original values

        Raises:
            TypeError: If an invalid type is provided for any attribute
            ValueError: If an invalid attribute name is provided

        Example:
            ```python
            new_config = config._create_modified_instance(
                pool_size=10,
                enable_wal_mode=False
            )
            ```
        """
        # Validate kwargs against allowed attributes
        valid_attrs: Set[str] = set(f.name for f in fields(self))
        for attr in kwargs:
            if attr not in valid_attrs:
                raise ValueError(
                    f"Invalid attribute: {attr}. Valid attributes are: {', '.join(valid_attrs)}"
                )

        # Type checking is handled by the replace() function and dataclass structure
        return replace(self, **kwargs)


# ==========================================
# Module Exports
# ==========================================

__all__ = [
    # Configuration class
    "DatabaseConfig",
    # Type definitions
    "DatabaseDialect",
    "TransactionIsolationLevel",
    "ConnectionPoolMode",
    "SQLitePragmas",
    "SQLTemplates",
    "PoolSettingsDict",
    "PragmaDict",
    # Error types
    "DatabaseConfigError",
]
