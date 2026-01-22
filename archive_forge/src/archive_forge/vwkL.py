"""
Image Interconversion GUI Database Module
=========================================

This module provides database functionalities essential for the Image Interconversion GUI application. It includes loading configurations, establishing database connections, executing queries, and managing image data with security measures like compression and encryption.

Dependencies:
- asyncio: For asynchronous programming.
- aiosqlite: Asynchronous SQLite database interaction.
- cryptography: For encryption and decryption of image data.
- PIL: For image processing tasks.

Setup/Initialization:
Ensure the following environment variables are set or a `config.ini` file is present:
- DATABASE_PATH: Path to the SQLite database file.
- KEY_FILE_PATH: Path to the encryption key file.

Author: Lloyd Handyside
Contact: ace1928@gmail.com
Version: 1.0.0
Creation Date: 2024-04-06
Last Modified: 2024-04-09
Last Reviewed: 2024-04-09
"""

import asyncio
import aiosqlite
from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv
from typing import Optional, List, Tuple, AsyncContextManager, NoReturn, AsyncGenerator
import backoff
from contextlib import asynccontextmanager
import io
from PIL import Image  # Import PIL.Image for image handling
import unittest
from unittest import IsolatedAsyncioTestCase

# Import ConfigManager, LoggingManager, EncryptionManager from core_services
from core_services import ConfigManager, LoggingManager, EncryptionManager
from image_processing import (
    validate_image,
    compress,
    decompress,
    ensure_image_format,
    encrypt,
    decrypt,
)

__all__ = [
    "load_database_configurations",
    "get_db_connection",
    "execute_db_query",
    "validate_pagination_params",
    "get_images_metadata",
    "insert_compressed_image",
    "retrieve_compressed_image",
    "migrate_schema",
    "init_db",
    "DatabaseConfig",
    "run_tests",
]


class DatabaseConfig:
    """
    Holds the configuration for the database connection and encryption key file path.

    Attributes:
        DB_PATH (str): Path to the SQLite database file.
        KEY_FILE (str): Path to the encryption key file.
        CONFIG_PATH: str = "config.ini"
    """

    DB_PATH: str = "image_db.sqlite"
    KEY_FILE: str = "encryption.key"
    CONFIG_PATH: str = "config.ini"


async def load_database_configurations():
    """
    Asynchronously loads database configurations from a config.ini file.

    Updates DatabaseConfig class attributes with values from the configuration file.

    Raises:
        FileNotFoundError: If the config.ini file does not exist.
        KeyError: If essential configuration keys are missing.
        Exception: For any other unexpected errors.

    Example:
        await load_database_configurations()
    """
    LoggingManager.debug("Starting to load database configurations.")
    try:
        config_manager = ConfigManager()
        config_path = os.path.join(
            os.path.dirname(__file__), DatabaseConfig.CONFIG_PATH
        )
        await config_manager.load_config(config_path, "Database", file_type="ini")
        DatabaseConfig.DB_PATH = config_manager.get(
            "Database", "db_path", "db_path", fallback="image_db.sqlite"
        )
        DatabaseConfig.KEY_FILE = config_manager.get(
            "Database", "key_file_path", "key_file_path", fallback="encryption.key"
        )
        LoggingManager.info("Database configurations loaded successfully.")
    except FileNotFoundError as e:
        LoggingManager.error(f"Configuration file not found: {e}")
        raise
    except KeyError as e:
        LoggingManager.error(f"Missing essential configuration key: {e}")
        raise
    except Exception as e:
        LoggingManager.error(f"Failed to load database configurations: {e}")
        raise


@asynccontextmanager
async def get_db_connection() -> AsyncContextManager[aiosqlite.Connection]:
    """
    Asynchronous context manager for managing database connections.

    This decorator ensures that the database connection is automatically opened at the start and properly closed after the block's execution, regardless of whether an exception occurred.

    Yields:
        aiosqlite.Connection: An open connection to the database.

    Raises:
        Exception: If connecting to the database fails.
    """
    LoggingManager.debug("Attempting to connect to the database.")
    try:
        database_connection = await aiosqlite.connect(DatabaseConfig.DB_PATH)
        LoggingManager.info("Database connection established successfully.")
        yield database_connection
    except Exception as e:
        LoggingManager.error(f"Failed to connect to the database: {e}")
        raise
    finally:
        await database_connection.close()
        LoggingManager.debug("Database connection closed.")


encryption_key = EncryptionManager.get_valid_encryption_key()
cipher_suite = Fernet(encryption_key)


@backoff.on_exception(
    backoff.expo,
    aiosqlite.OperationalError,
    max_time=9,
    on_backoff=lambda details: LoggingManager.warning(
        f"Retrying due to: {details['exception']}"
    ),
)
# Implements exponential backoff retry for operational errors in database queries.
# Retries for up to 9 seconds before giving up, logging warnings on each retry.
async def execute_db_query(query: str, parameters: tuple = ()) -> aiosqlite.Cursor:
    """
    Executes a database query asynchronously.

    This function executes a given SQL query with the provided parameters. It uses an exponential backoff strategy for retries in case of operational errors.

    Parameters:
        query (str): The SQL query to execute.
        parameters (tuple): Parameters for the SQL query.

    Returns:
        aiosqlite.Cursor: The cursor resulting from the query execution.

    Raises:
        aiosqlite.OperationalError: For operational database errors.
        Exception: For any other unexpected errors.
    """
    LoggingManager.debug(
        f"Executing database query: {query} with parameters: {parameters}"
    )
    try:
        async with get_db_connection() as db_connection:
            cursor = await db_connection.execute(query, parameters)
            await db_connection.commit()
            LoggingManager.info(f"Query executed successfully: {query}")
            return cursor
    except aiosqlite.OperationalError as e:
        LoggingManager.error(f"Database operational error: {e}, Query: {query}")
        raise
    except Exception as e:
        LoggingManager.error(
            f"Unexpected error executing database query: {e}, Query: {query}"
        )
        raise


def validate_pagination_params(offset: int, limit: int) -> NoReturn:
    """
    Validates the pagination parameters for database queries.

    Ensures that the offset is non-negative and the limit is positive, raising appropriate exceptions if not.

    Parameters:
        offset (int): The offset from where to start fetching the records.
        limit (int): The maximum number of records to fetch.

    Raises:
        TypeError: If either offset or limit is not an integer.
        ValueError: If offset is negative or limit is not positive.
    """
    if not isinstance(offset, int) or not isinstance(limit, int):
        raise TypeError("Offset and limit must be integers.")
    if offset < 0 or limit <= 0:
        raise ValueError("Offset must be non-negative and limit must be positive.")


async def get_images_metadata(
    offset: int = 0, limit: int = 10
) -> List[Tuple[str, str]]:
    """
    Retrieves metadata for images stored in the database within specified pagination parameters.

    This function fetches the hash and format for images stored in the database, limited by the provided offset and limit parameters for pagination purposes.

    Parameters:
        offset (int): The offset from where to start fetching the records. Defaults to 0.
        limit (int): The maximum number of records to fetch. Defaults to 10.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the hash and format of the images.

    Raises:
        Exception: If retrieving the metadata fails.
    """
    validate_pagination_params(offset, limit)
    query = "SELECT hash, format FROM images LIMIT ? OFFSET ?"
    async with get_db_connection() as db_connection:
        cursor = await db_connection.execute(query, (limit, offset))
        result = await cursor.fetchall()
    return result


async def insert_compressed_image(hash: str, format: str, data: bytes) -> None:
    """
    Inserts a compressed and encrypted image into the database.

    This function ensures the image format, validates the image, compresses, and encrypts the data before inserting it into the database.

    Parameters:
        hash (str): The unique hash identifier for the image.
        format (str): The format of the image (e.g., 'png', 'jpg').
        data (bytes): The raw image data to be compressed and encrypted.

    Raises:
        Exception: If the image fails to be inserted into the database.
        ValueError: If the image data fails validation.
    """
    LoggingManager.debug(f"Inserting compressed image with hash: {hash}")
    try:
        image = ensure_image_format(data)
        if not validate_image(image):
            LoggingManager.error(f"Data validation failed for image {hash}.")
            raise ValueError(f"Data validation failed for image {hash}.")
        compressed_data = await compress(data, format)
        encrypted_data = await encrypt(compressed_data)
        query = "INSERT OR REPLACE INTO images (hash, format, compressed_data) VALUES (?, ?, ?)"
        await execute_db_query(query, (hash, format, encrypted_data))
        LoggingManager.info(f"Image {hash} inserted successfully.")
    except Exception as e:
        LoggingManager.error(f"Failed to insert image {hash}: {e}")
        raise


async def retrieve_compressed_image(hash: str) -> Optional[tuple]:
    """
    Retrieves a compressed and encrypted image from the database.

    This function fetches the compressed and encrypted image data from the database using the provided hash, decrypts, and decompresses it before returning.

    Parameters:
        hash (str): The unique hash identifier for the image to retrieve.

    Returns:
        Optional[tuple]: A tuple containing the decompressed image data and its format, or None if the image is not found.

    Raises:
        Exception: If retrieving the image fails.
    """
    LoggingManager.debug(f"Retrieving compressed image with hash: {hash}")
    query = "SELECT compressed_data, format FROM images WHERE hash = ?"
    try:
        async with get_db_connection() as db_connection:
            cursor = await db_connection.execute(query, (hash,))
            result = await cursor.fetchone()
            if result:
                compressed_data, format = result
                decrypted_data = await decrypt(compressed_data)
                data = await decompress(decrypted_data)
                LoggingManager.info(f"Image {hash} retrieved successfully.")
                return data, format
            LoggingManager.warning(f"No image found with hash: {hash}")
            return None
    except Exception as e:
        LoggingManager.error(f"Failed to retrieve image {hash}: {e}")
        raise


async def migrate_schema():
    """
    Migrates the database schema to the latest version.

    This function checks the current database schema against the expected schema and applies any necessary migrations to bring it up to date.

    Raises:
        Exception: If schema migration fails.
    """
    expected_schema = {
        "images": [
            ("hash", "TEXT PRIMARY KEY"),
            ("format", "TEXT NOT NULL"),
            ("compressed_data", "BLOB NOT NULL"),
        ]
    }

    async with get_db_connection() as db_connection:
        for table, expected_columns in expected_schema.items():
            # Check if table exists
            cursor = await db_connection.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';"
            )
            if not await cursor.fetchone():
                # Create table if it doesn't exist
                columns = ", ".join([f"{col[0]} {col[1]}" for col in expected_columns])
                create_table_query = f"CREATE TABLE {table} ({columns})"
                await db_connection.execute(create_table_query)
                continue  # Skip to next iteration since table was just created

            # Proceed with adding missing columns if table exists
            cursor = await db_connection.execute(f"PRAGMA table_info({table})")
            current_columns = {info[1]: info[2] for info in await cursor.fetchall()}
            missing_columns = [
                col for col in expected_columns if col[0] not in current_columns
            ]

            for col_name, col_type in missing_columns:
                alter_query = f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                await db_connection.execute(alter_query)

        await db_connection.commit()
        LoggingManager.info("Database schema migration completed successfully.")


async def init_db():
    """
    Initializes the database and migrates the schema to the latest version.

    This function ensures that the database schema is up to date by calling the migrate_schema function. It is intended to be run at application startup to prepare the database for use.

    Raises:
        Exception: If initializing the database fails.
    """
    async with get_db_connection() as db_connection:
        try:
            await migrate_schema()
            LoggingManager.info(
                "Database initialization and schema migration completed."
            )
        except Exception as e:
            LoggingManager.error(f"Failed to initialize database: {e}")
            raise


async def initialize_test_database():
    async with aiosqlite.connect(DatabaseConfig.DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                hash TEXT PRIMARY KEY,
                format TEXT NOT NULL,
                compressed_data BLOB NOT NULL
            )
        """
        )
        await db.commit()


class BaseDatabaseTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await initialize_test_database()

    async def asyncTearDown(self):
        async with aiosqlite.connect(DatabaseConfig.DB_PATH) as db:
            await db.execute("DROP TABLE IF EXISTS images")
            await db.commit()


class TestDatabaseConfigurations(BaseDatabaseTestCase):
    async def test_load_database_configurations_failure(self):
        original_path = DatabaseConfig.CONFIG_PATH
        DatabaseConfig.CONFIG_PATH = "/tmp/nonexistent_path.ini"
        with self.assertRaises(FileNotFoundError):
            await load_database_configurations()
        DatabaseConfig.CONFIG_PATH = original_path


class TestDatabaseConnection(BaseDatabaseTestCase):
    async def test_get_db_connection_failure(self):
        # Temporarily set an invalid database path to simulate connection failure
        original_db_path = DatabaseConfig.DB_PATH
        DatabaseConfig.DB_PATH = "invalid/path/to/database.db"
        with self.assertRaises(Exception):
            async with get_db_connection() as _:
                pass
        # Restore the original path after the test
        DatabaseConfig.DB_PATH = original_db_path


class TestExecuteDBQuery(BaseDatabaseTestCase):
    async def test_execute_db_query_success(self):
        query = (
            "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)"
        )
        await execute_db_query(query)
        # Further assertions can be made to check if the table exists

    async def test_execute_db_query_failure(self):
        query = "INVALID SQL QUERY"
        with self.assertRaises(Exception):
            await execute_db_query(query)


class TestPaginationValidation(unittest.TestCase):
    def test_validate_pagination_params_success(self):
        validate_pagination_params(0, 10)  # Should pass without exception

    def test_validate_pagination_params_failure(self):
        with self.assertRaises(ValueError):
            validate_pagination_params(-1, 10)


class TestImageMetadata(BaseDatabaseTestCase):
    async def test_get_images_metadata_success(self):
        # Assuming the database and images table are properly set up
        metadata = await get_images_metadata()
        self.assertIsInstance(metadata, list)
        # Further assertions depending on expected metadata format


class TestImageInsertion(BaseDatabaseTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        # Insert known image data for retrieval test
        with open("test_image.png", "rb") as image_file:
            image_data = image_file.read()
        await insert_compressed_image("test_hash", "png", image_data)

    async def test_insert_compressed_image_success(self):
        # Assuming valid hash, format, and data
        with open("test_image.png", "rb") as image_file:
            image_data = image_file.read()
        await insert_compressed_image("hash123", "png", image_data)
        # Further assertions to check if the image was inserted correctly

    async def test_insert_compressed_image_failure(self):
        # Assuming invalid data to cause failure
        with self.assertRaises(ValueError):
            await insert_compressed_image("hash123", "jpg", b"")


class TestImageRetrieval(BaseDatabaseTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        with open("test_image.png", "rb") as image_file:
            image_data = image_file.read()
        # Insert known image data for retrieval test
        await insert_compressed_image("test_hash", "png", image_data)

    async def test_retrieve_compressed_image_success(self):
        # Now retrieving an image that we know exists
        result = await retrieve_compressed_image("test_hash")
        self.assertIsNotNone(result)
        # Further assertions depending on the expected result format

    async def test_retrieve_compressed_image_failure(self):
        # Assuming no image with the given hash exists
        result = await retrieve_compressed_image("non_existent_hash")
        self.assertIsNone(result)


class TestSchemaMigration(BaseDatabaseTestCase):
    async def test_migrate_schema_success(self):
        # Step 1: Ensure the database is in a known state, lacking the expected schema.
        async with get_db_connection() as db_connection:
            await db_connection.execute("DROP TABLE IF EXISTS images")
            await db_connection.commit()

        # Step 2: Run the migrate_schema function to apply necessary migrations.
        await migrate_schema()

        # Step 3: Verify that the database schema includes the expected tables and columns.
        async with get_db_connection() as db_connection:
            cursor = await db_connection.execute("PRAGMA table_info(images)")
            columns_info = await cursor.fetchall()
            columns_names = [info[1] for info in columns_info]  # Extract column names

            # Define the expected columns in the 'images' table.
            expected_columns = {"hash", "format", "compressed_data"}

            # Verify that all expected columns are present in the table.
            self.assertTrue(
                expected_columns.issubset(set(columns_names)),
                "The 'images' table does not contain all the expected columns.",
            )

        LoggingManager.info("Schema migration test passed successfully.")


class TestDatabaseInitialization(BaseDatabaseTestCase):
    async def test_init_db_success(self):
        await init_db()
        # Further assertions to check if the database was initialized correctly

    async def test_init_db_failure(self):
        # Backup the original database path
        original_db_path = DatabaseConfig.DB_PATH
        # Set to an invalid path to simulate initialization failure
        DatabaseConfig.DB_PATH = "/invalid/path/to/database.db"
        with self.assertRaises(Exception):
            await init_db()
        # Restore the original database path
        DatabaseConfig.DB_PATH = original_db_path


if __name__ == "__main__":
    asyncio.run(unittest.main())
