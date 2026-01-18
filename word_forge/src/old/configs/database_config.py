"""
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

from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Dict, TypedDict

from word_forge.configs.config_essentials import (
    DATA_ROOT,
    ConnectionPoolMode,
    DatabaseConfigError,
    DatabaseDialect,
    PathLike,
    SQLitePragmas,
    SQLTemplates,
    TransactionIsolationLevel,
)
from word_forge.configs.config_types import EnvMapping


# More precise type definitions
class PoolSettingsDict(TypedDict):
    """Type definition for connection pool settings."""

    mode: ConnectionPoolMode
    size: int
    timeout: float
    recycle: int


# Type aliases for clearer return signatures
PragmaDict = Dict[str, str]


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
        default_factory=lambda: {
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
        }
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

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_DB_PATH": ("db_path", str),
        "WORD_FORGE_DB_DIALECT": ("dialect", DatabaseDialect),
        "WORD_FORGE_DB_POOL_SIZE": ("pool_size", int),
        "WORD_FORGE_DB_POOL_MODE": ("pool_mode", str),
        "WORD_FORGE_DB_ISOLATION": ("isolation_level", str),
        "WORD_FORGE_DB_FOREIGN_KEYS": ("enable_foreign_keys", bool),
    }

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
        result = dict(self.pragmas)

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
        read_pragmas = dict(self.pragmas)
        read_pragmas.update(
            {
                "synchronous": "NORMAL",  # Less durability, more speed
                "journal_mode": "WAL",  # Allow concurrent reads/writes
                "cache_size": str(-8000),  # Larger cache for read-heavy workloads
                "temp_store": "MEMORY",  # Keep temp tables in memory
            }
        )

        return self._create_modified_instance(
            pragmas=read_pragmas,
            pool_size=max(self.pool_size, 8),  # Larger pool for more concurrent reads
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
        write_pragmas = dict(self.pragmas)
        write_pragmas.update(
            {
                "synchronous": "NORMAL",  # Balance between durability and performance
                "journal_mode": "WAL",  # Better for writes than DELETE
                "cache_size": str(-4000),  # Moderate cache
                "mmap_size": str(1073741824),  # 1GB memory map
            }
        )

        return self._create_modified_instance(
            pragmas=write_pragmas,
            pool_size=max(self.pool_size, 4),  # Moderate pool size for writes
            pool_recycle=1800,  # More frequent recycling for write connections
            enable_wal_mode=True,  # WAL is also good for writes
            page_size=8192,  # Larger page size for bulk writes
            cache_size=4000,  # Moderate cache size
        )

    def validate(self) -> None:
        """
        Validate the entire configuration for consistency and correctness.

        Checks all configuration parameters for valid values and
        compatibility with the selected database dialect.

        Raises:
            DatabaseConfigError: If any validation fails

        Example:
            ```python
            try:
                config.database.validate()
                print("Configuration is valid")
            except DatabaseConfigError as e:
                print(f"Invalid configuration: {e}")
            ```
        """
        errors = []

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
                errors.append(f"Pool timeout must be positive, got {self.pool_timeout}")

        if errors:
            raise DatabaseConfigError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )

    # ==========================================
    # Private Helper Methods
    # ==========================================

    def _create_modified_instance(self, **kwargs) -> "DatabaseConfig":
        """
        Create a new configuration instance with modified attributes.

        Args:
            **kwargs: Attribute name-value pairs to override

        Returns:
            DatabaseConfig: New instance with specified modifications
        """
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
