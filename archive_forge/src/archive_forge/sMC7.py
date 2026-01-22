import asyncio
import aiosqlite
from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv
from typing import Optional, List, Tuple
import backoff
from contextlib import asynccontextmanager
import io
from PIL import Image  # Import PIL.Image for image handling

# Import ConfigManager, LoggingManager, EncryptionManager from core_services
from core_services import ConfigManager, LoggingManager, EncryptionManager
from image_processing import (
    validate_image,
    compress,
    decompress,
    ensure_image_format,
)


class DatabaseConfig:
    DB_PATH: str = "image_db.sqlite"
    KEY_FILE: str = "encryption.key"


async def load_database_configurations():
    LoggingManager.debug("Starting to load database configurations.")
    try:
        config_manager = ConfigManager()
        config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        await config_manager.load_config(config_path, "Database", file_type="ini")
        DatabaseConfig.DB_PATH = config_manager.get(
            "Database", "db_path", "db_path", fallback="image_db.sqlite"
        )
        DatabaseConfig.KEY_FILE = config_manager.get(
            "Database", "key_file_path", "key_file_path", fallback="encryption.key"
        )
        LoggingManager.info("Database configurations loaded successfully.")
    except Exception as e:
        LoggingManager.error(f"Failed to load database configurations: {e}")
        raise


@asynccontextmanager
async def get_db_connection():
    LoggingManager.debug("Attempting to connect to the database.")
    try:
        conn = await aiosqlite.connect(DatabaseConfig.DB_PATH)
        LoggingManager.debug("Database connection established.")
        yield conn
    except Exception as e:
        LoggingManager.error(f"Failed to connect to the database: {e}")
        raise
    finally:
        await conn.close()
        LoggingManager.debug("Database connection closed.")


encryption_key = EncryptionManager.get_valid_encryption_key()
cipher_suite = Fernet(encryption_key)


@backoff.on_exception(
    backoff.expo,
    aiosqlite.OperationalError,
    max_time=60,
    on_backoff=lambda details: LoggingManager.warning(
        f"Retrying due to: {details['exception']}"
    ),
)
async def execute_db_query(query: str, parameters: tuple = ()) -> aiosqlite.Cursor:
    LoggingManager.debug(
        f"Executing database query: {query} with parameters: {parameters}"
    )
    try:
        async with get_db_connection() as db:
            cursor = await db.execute(query, parameters)
            await db.commit()
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


def validate_pagination_params(offset: int, limit: int):
    if not isinstance(offset, int) or not isinstance(limit, int):
        raise TypeError("Offset and limit must be integers.")
    if offset < 0 or limit <= 0:
        raise ValueError("Offset must be non-negative and limit must be positive.")


async def get_images_metadata(
    offset: int = 0, limit: int = 10
) -> List[Tuple[str, str]]:
    validate_pagination_params(offset, limit)
    query = "SELECT hash, format FROM images LIMIT ? OFFSET ?"
    async with get_db_connection() as db:
        cursor = await db.execute(query, (limit, offset))
        result = await cursor.fetchall()
    return result


async def insert_compressed_image(hash: str, format: str, data: bytes) -> None:
    LoggingManager.debug(f"Inserting compressed image with hash: {hash}")
    try:
        image = ensure_image_format(data)
        if not validate_image(image):
            LoggingManager.error(f"Data validation failed for image {hash}.")
            return
        compressed_data = await compress(data, format)
        encrypted_data = EncryptionManager.encrypt(
            compressed_data
        )  # Utilizing EncryptionManager
        query = "INSERT OR REPLACE INTO images (hash, format, compressed_data) VALUES (?, ?, ?)"
        await execute_db_query(query, (hash, format, encrypted_data))
        LoggingManager.info(f"Image {hash} inserted successfully.")
    except Exception as e:
        LoggingManager.error(f"Failed to insert image {hash}: {e}")
        raise


async def retrieve_compressed_image(hash: str) -> Optional[tuple]:
    LoggingManager.debug(f"Retrieving compressed image with hash: {hash}")
    query = "SELECT compressed_data, format FROM images WHERE hash = ?"
    try:
        async with get_db_connection() as db:
            cursor = await db.execute(query, (hash,))
            result = await cursor.fetchone()
            if result:
                compressed_data, format = result
                decrypted_data = EncryptionManager.decrypt(
                    compressed_data
                )  # Utilizing EncryptionManager
                data = decompress(decrypted_data, format)
                LoggingManager.info(f"Image {hash} retrieved successfully.")
                return data, format
            LoggingManager.warning(f"No image found with hash: {hash}")
            return None
    except Exception as e:
        LoggingManager.error(f"Failed to retrieve image {hash}: {e}")
        raise


async def migrate_schema():
    expected_schema = {
        "images": [
            ("hash", "TEXT PRIMARY KEY"),
            ("format", "TEXT NOT NULL"),
            ("compressed_data", "BLOB NOT NULL"),
        ]
    }

    async with get_db_connection() as db:
        for table, expected_columns in expected_schema.items():
            cursor = await db.execute(f"PRAGMA table_info({table})")
            current_columns = {info[1]: info[2] for info in await cursor.fetchall()}
            missing_columns = [
                col for col in expected_columns if col[0] not in current_columns
            ]

            for col_name, col_type in missing_columns:
                alter_query = f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                await db.execute(alter_query)
                LoggingManager.info(
                    f"Added missing column '{col_name}' to '{table}' table."
                )

        await db.commit()
        LoggingManager.info("Database schema migration completed successfully.")


async def init_db():
    async with get_db_connection() as db:
        try:
            await migrate_schema()
            LoggingManager.info(
                "Database initialization and schema migration completed."
            )
        except Exception as e:
            LoggingManager.error(f"Failed to initialize database: {e}")
            raise


async def main():
    await load_database_configurations()
    # Additional initialization or operations can be added here


# Test suite implementation
async def run_tests():
    """
    Runs a suite of tests to verify the functionality of the database operations.
    """
    # Test database connection
    async with get_db_connection() as conn:
        assert conn is not None, "Database connection failed."

    # Test schema migration
    await migrate_schema()
    LoggingManager.info("Schema migration test passed.")

    # Test insert and retrieve compressed image
    test_hash = "test_hash"
    test_format = "png"
    test_image_path = (
        "/home/lloyd/EVIE/scripts/image_interconversion_gui/test_image.png"
    )
    with open(test_image_path, "rb") as image_file:
        test_data = image_file.read()

    await insert_compressed_image(test_hash, test_format, test_data)
    retrieved = await retrieve_compressed_image(test_hash)
    assert retrieved is not None, "Failed to retrieve compressed image."
    retrieved_data, retrieved_format = retrieved
    assert retrieved_format == test_format, "Retrieved format does not match."
    LoggingManager.info("Insert and retrieve compressed image test passed.")

    # Test get images metadata with pagination
    metadata = await get_images_metadata(0, 1)
    assert (
        isinstance(metadata, list) and len(metadata) <= 1
    ), "Get images metadata test failed."
    LoggingManager.info("Get images metadata test passed.")

    # Test validate pagination params
    try:
        validate_pagination_params(-1, 10)
    except ValueError:
        LoggingManager.info("Validate pagination params test passed.")
    else:
        assert False, "Validate pagination params test failed."

    LoggingManager.info("All tests passed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(run_tests())
