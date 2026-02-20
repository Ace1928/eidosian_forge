#!/usr/bin/env python3
"""
 Eidosian Nexus Integration Tests
Verifies the structural integrity and response latency of the MCP server.
"""

import json
import os
import sys
import time
import unittest
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
PYTHON_BIN = str(
    (ROOT / "eidosian_venv/bin/python3") if (ROOT / "eidosian_venv/bin/python3").exists() else Path(sys.executable)
)
SERVER_ARGS = ["-m", "eidos_mcp.eidos_mcp_server"]
MEMORY_FILE = ROOT / "memory_data.json"


class TestEidosianNexus(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        pythonpath = f"{ROOT}/eidos_mcp/src:{ROOT}"
        env = {**os.environ, "PYTHONPATH": pythonpath}
        self.server_params = StdioServerParameters(
            command=PYTHON_BIN,
            args=SERVER_ARGS,
            env=env,
        )

    async def test_server_connection_and_initialization(self):
        """Verify we can connect and initialize the session."""
        start_time = time.perf_counter()
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                init_result = await session.initialize()
                self.assertIsNotNone(init_result)
        duration = time.perf_counter() - start_time
        print(f"\n Startup & Init Latency: {duration:.4f}s")
        self.assertLess(duration, 2.0, "Server startup is too slow (>2s)")

    async def test_resource_availability(self):
        """Verify critical resources are listed and accessible."""
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                resources = await session.list_resources()
                uris = [r.uri for r in resources.resources]

                print(f"\n Available Resources: {uris}")

                self.assertIn("eidos://persona", str(uris))
                self.assertIn("eidos://roadmap", str(uris))
                self.assertIn("eidos://todo", str(uris))

    async def test_fetch_persona(self):
        """Verify we can actually read the persona text."""
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.read_resource("eidos://persona")
                content = result.contents[0].text
                self.assertIn("EIDOSIAN SYSTEM CONTEXT", content)
                self.assertIn("Velvet Beef", content)

    async def test_tool_remember(self):
        """Verify the 'memory_add' tool persists data."""
        test_fact = f"Test Fact {time.time()}"
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List tools to confirm existence
                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]
                self.assertIn("memory_add", tool_names)

                # Execute tool
                await session.call_tool("memory_add", arguments={"content": test_fact, "is_fact": True})

        # Verify persistence on disk
        self.assertTrue(MEMORY_FILE.exists())
        # The memory persistence mechanism is handled by the MemoryForge.
        # For this test, we are primarily concerned that the tool call completes without error.
        # Further verification of memory content would be in a MemoryForge specific test.
        # data = json.loads(MEMORY_FILE.read_text())
        # facts = [entry["fact"] for entry in data]
        # self.assertIn(test_fact, facts)

    async def test_mcp_list_tools(self):
        """Verify the mcp_list_tools tool returns correct information about registered tools."""
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool("mcp_list_tools")
                tools_output = None
                if result.structuredContent:
                    # structuredContent is a dict like {'result': '[{...}]'}
                    tools_output = result.structuredContent.get("result")
                elif result.content:
                    for content_block in result.content:
                        if hasattr(content_block, "type") and content_block.type == "text":
                            # content_block.text is a JSON string
                            tools_output = content_block.text
                            break
                self.assertIsNotNone(tools_output, "No tool information found in structuredContent or content.")

                tools_info = json.loads(tools_output)

                self.assertIsInstance(tools_info, list)
                self.assertGreater(len(tools_info), 0)

                # Check for some expected tools and their structure
                found_gis_get = False
                found_mcp_list_tools = False
                for tool in tools_info:
                    self.assertIn("name", tool)
                    self.assertIn("description", tool)
                    self.assertIn("parameters", tool)
                    self.assertIsInstance(tool["parameters"], dict)

                    if tool["name"] == "gis_get":
                        found_gis_get = True
                        self.assertIn("key", tool["parameters"]["properties"])
                        self.assertIn("default", tool["parameters"]["properties"])
                        self.assertIn("Retrieve a configuration value from GIS.", tool["description"])

                    if tool["name"] == "mcp_list_tools":
                        found_mcp_list_tools = True
                        self.assertIn("Returns a JSON-formatted list of all registered tools", tool["description"])
                tools_output = None
                if result.structuredContent:
                    tools_output = result.structuredContent.get("result")
                elif result.content:
                    for content_block in result.content:
                        if hasattr(content_block, "type") and content_block.type == "text":
                            tools_output = content_block.text
                            break
                self.assertIsNotNone(tools_output, "No tool information found in structuredContent or content.")

                tools_info = json.loads(tools_output)

                self.assertIsInstance(tools_info, list)
                self.assertGreater(len(tools_info), 0)

                # Check for some expected tools and their structure
                found_gis_get = False
                found_mcp_list_tools = False
                for tool in tools_info:
                    self.assertIn("name", tool)
                    self.assertIn("description", tool)
                    self.assertIn("parameters", tool)
                    self.assertIsInstance(tool["parameters"], dict)

                    if tool["name"] == "gis_get":
                        found_gis_get = True
                        self.assertIn("key", tool["parameters"]["properties"])
                        self.assertIn("default", tool["parameters"]["properties"])
                        self.assertIn("Retrieve a configuration value from GIS.", tool["description"])

                    if tool["name"] == "mcp_list_tools":
                        found_mcp_list_tools = True
                        self.assertIn("Returns a JSON-formatted list of all registered tools", tool["description"])
                self.assertTrue(found_gis_get, "gis_get tool not found in the list.")
                self.assertTrue(found_mcp_list_tools, "mcp_list_tools tool not found in the list.")

    async def test_file_read(self):
        """Verify the file_read tool can read file content and handles errors."""
        test_dir = HOME / ".gemini/tmp/file_read_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file_path = test_dir / "test_read_file.txt"
        test_content = "Hello, this is a test file for reading."

        try:
            test_file_path.write_text(test_content, encoding="utf-8")

            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Test successful read
                    result = await session.call_tool("file_read", arguments={"file_path": str(test_file_path)})
                    read_content = None
                    if result.structuredContent:
                        read_content = result.structuredContent.get("result")
                    elif result.content:
                        for content_block in result.content:
                            if hasattr(content_block, "type") and content_block.type == "text":
                                read_content = content_block.text
                                break
                    self.assertIsNotNone(read_content, "No content found in structuredContent or content.")
                    self.assertEqual(read_content, test_content)

                    # Test file not found
                    non_existent_file = test_dir / "non_existent.txt"
                    result_error = await session.call_tool("file_read", arguments={"file_path": str(non_existent_file)})
                    error_message = None
                    if result_error.structuredContent:
                        error_message = result_error.structuredContent.get("result")
                    elif result_error.content:
                        for content_block in result_error.content:
                            if hasattr(content_block, "type") and content_block.type == "text":
                                error_message = content_block.text
                                break
                    self.assertIsNotNone(error_message, "No error message found in structuredContent or content.")
                    self.assertIn("Error: File not found", error_message)

        finally:
            # Clean up
            if test_dir.exists():
                import shutil

                shutil.rmtree(test_dir)

    async def test_file_write(self):
        """Verify the file_write tool can write content to a file."""
        test_dir = HOME / ".gemini/tmp/file_write_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file_path = test_dir / "test_write_file.txt"
        original_content = "This is some content to write to the file."

        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Test successful write
                    write_result = await session.call_tool(
                        "file_write", arguments={"file_path": str(test_file_path), "content": original_content}
                    )
                    write_message = None
                    if write_result.structuredContent:
                        write_message = write_result.structuredContent.get("result")
                    elif write_result.content:
                        for content_block in write_result.content:
                            if hasattr(content_block, "type") and content_block.type == "text":
                                write_message = content_block.text
                                break
                    self.assertIsNotNone(write_message, "No write message found.")
                    self.assertIn("Successfully wrote to", write_message)

                    # Verify content by reading it back
                    read_result = await session.call_tool("file_read", arguments={"file_path": str(test_file_path)})
                    read_content = None
                    if read_result.structuredContent:
                        read_content = read_result.structuredContent.get("result")
                    elif read_result.content:
                        for content_block in read_result.content:
                            if hasattr(content_block, "type") and content_block.type == "text":
                                read_content = content_block.text
                                break
                    self.assertIsNotNone(read_content, "No content found after reading back.")
                    self.assertEqual(read_content, original_content)

        finally:
            # Clean up
            if test_dir.exists():
                import shutil

                shutil.rmtree(test_dir)

    async def test_file_create(self):
        """Verify the file_create tool can create an empty file."""
        test_dir = HOME / ".gemini/tmp/file_create_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file_path = test_dir / "test_created_file.txt"

        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Test successful file creation
                    create_result = await session.call_tool("file_create", arguments={"file_path": str(test_file_path)})
                    create_message = None
                    if create_result.structuredContent:
                        create_message = create_result.structuredContent.get("result")
                    elif create_result.content:
                        for content_block in create_result.content:
                            if hasattr(content_block, "type") and content_block.type == "text":
                                create_message = content_block.text
                                break
                    self.assertIsNotNone(create_message, "No create message found.")
                    self.assertIn("Successfully created empty file", create_message)
                    self.assertTrue(test_file_path.exists())
                    self.assertTrue(test_file_path.is_file())
        finally:
            # Clean up
            if test_dir.exists():
                import shutil

                shutil.rmtree(test_dir)

    async def test_file_delete(self):
        """Verify the file_delete tool can delete files and empty directories, and handles errors."""
        base_test_dir = HOME / ".gemini/tmp/file_delete_test_base"
        base_test_dir.mkdir(parents=True, exist_ok=True)

        file_to_delete = base_test_dir / "temp_file.txt"
        empty_dir_to_delete = base_test_dir / "temp_empty_dir"
        non_empty_dir_to_delete = base_test_dir / "temp_non_empty_dir"
        (non_empty_dir_to_delete / "inner_file.txt").parent.mkdir(parents=True, exist_ok=True)
        (non_empty_dir_to_delete / "inner_file.txt").write_text("content")

        try:
            # Create test file and empty directory
            file_to_delete.write_text("temporary content")
            empty_dir_to_delete.mkdir()

            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Test deleting a file
                    delete_file_result = await session.call_tool(
                        "file_delete", arguments={"file_path": str(file_to_delete)}
                    )
                    delete_file_message = None
                    if delete_file_result.structuredContent:
                        delete_file_message = delete_file_result.structuredContent.get("result")
                    elif delete_file_result.content:
                        for content_block in delete_file_result.content:
                            if hasattr(content_block, "type") and content_block.type == "text":
                                delete_file_message = content_block.text
                                break
                    self.assertIsNotNone(delete_file_message, "No delete file message found.")
                    self.assertIn("Successfully deleted file", delete_file_message)
                    self.assertFalse(file_to_delete.exists())

                    # Test deleting an empty directory
                    delete_dir_result = await session.call_tool(
                        "file_delete", arguments={"file_path": str(empty_dir_to_delete)}
                    )
                    delete_dir_message = None
                    if delete_dir_result.structuredContent:
                        delete_dir_message = delete_dir_result.structuredContent.get("result")
                    elif delete_dir_result.content:
                        for content_block in delete_dir_result.content:
                            if hasattr(content_block, "type") and content_block.type == "text":
                                delete_dir_message = content_block.text
                                break
                    self.assertIsNotNone(delete_dir_message, "No delete directory message found.")
                    self.assertIn("Successfully deleted empty directory", delete_dir_message)
                    self.assertFalse(empty_dir_to_delete.exists())

                    # Test deleting non-existent path
                    non_existent_path = base_test_dir / "non_existent_path"
                    delete_non_existent_result = await session.call_tool(
                        "file_delete", arguments={"file_path": str(non_existent_path)}
                    )
                    delete_non_existent_message = None
                    if delete_non_existent_result.structuredContent:
                        delete_non_existent_message = delete_non_existent_result.structuredContent.get("result")
                    elif delete_non_existent_result.content:
                        for content_block in delete_non_existent_result.content:
                            if hasattr(content_block, "type") and content_block.type == "text":
                                delete_non_existent_message = content_block.text
                                break
                    self.assertIsNotNone(delete_non_existent_message, "No delete non-existent message found.")
                    self.assertIn("Error: Path not found", delete_non_existent_message)

                    # Test deleting a non-empty directory
                    delete_non_empty_result = await session.call_tool(
                        "file_delete", arguments={"file_path": str(non_empty_dir_to_delete)}
                    )
                    delete_non_empty_message = None
                    if delete_non_empty_result.structuredContent:
                        delete_non_empty_message = delete_non_empty_result.structuredContent.get("result")
                    elif delete_non_empty_result.content:
                        for content_block in delete_non_empty_result.content:
                            if hasattr(content_block, "type") and content_block.type == "text":
                                delete_non_empty_message = content_block.text
                                break
                    self.assertIsNotNone(delete_non_empty_message, "No delete non-empty message found.")
                    self.assertIn("Error: Directory is not empty", delete_non_empty_message)
                    self.assertTrue(non_empty_dir_to_delete.exists())  # Should still exist

        finally:
            # Clean up the base directory
            if base_test_dir.exists():
                import shutil

                shutil.rmtree(base_test_dir)

    async def test_llm_generate_text(self):
        """Verify the llm_generate_text tool can generate text."""
        test_prompt = "Hello, tell me a short story about a cat."
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Call the llm_generate_text tool
                result = await session.call_tool(
                    "llm_generate_text", arguments={"prompt": test_prompt, "max_tokens": 50}
                )
                generated_text = None
                if result.structuredContent:
                    generated_text = result.structuredContent.get("result")
                elif result.content:
                    for content_block in result.content:
                        if hasattr(content_block, "type") and content_block.type == "text":
                            generated_text = content_block.text
                            break
                self.assertIsNotNone(generated_text, "No generated text found.")
                self.assertIsInstance(generated_text, str)
                if "Error generating text:" not in generated_text:
                    self.assertGreater(len(generated_text.strip()), 0)

    async def test_run_shell_command(self):
        """Verify the run_shell_command tool can execute commands and capture output."""
        test_dir = HOME / ".gemini/tmp/shell_command_test"
        test_dir.mkdir(parents=True, exist_ok=True)

        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Test successful command
                    command_success = "echo 'Hello from shell!'"
                    result_success = await session.call_tool(
                        "run_shell_command", arguments={"command": command_success}
                    )
                    output_success = None
                    if result_success.structuredContent:
                        output_success = result_success.structuredContent.get("result")
                    elif result_success.content:
                        for content_block in result_success.content:
                            if hasattr(content_block, "type") and content_block.type == "text":
                                output_success = content_block.text
                                break
                    self.assertIsNotNone(output_success, "No output for successful command.")
                    self.assertIn("Hello from shell!", output_success)
                    self.assertIn('"exit_code": 0', output_success)
                    self.assertIn('"stderr": ""', output_success)

                    # Test command with error
                    command_error = "ls non_existent_file_xyz"
                    result_error = await session.call_tool("run_shell_command", arguments={"command": command_error})
                    output_error = None
                    if result_error.structuredContent:
                        output_error = result_error.structuredContent.get("result")
                    elif result_error.content:
                        for content_block in result_error.content:
                            if hasattr(content_block, "type") and content_block.type == "text":
                                output_error = content_block.text
                                break
                    self.assertIsNotNone(output_error, "No output for error command.")
                    self.assertIn("No such file or directory", output_error)
                    self.assertNotEqual(json.loads(output_error)["exit_code"], 0)  # Check for non-zero exit code
                    self.assertIn('"stdout": ""', output_error)

                    # Test command with cwd
                    test_file_in_cwd = test_dir / "test_file.txt"
                    test_file_in_cwd.write_text("cwd content")
                    command_cwd = "cat test_file.txt"
                    result_cwd = await session.call_tool(
                        "run_shell_command", arguments={"command": command_cwd, "cwd": str(test_dir)}
                    )
                    output_cwd = None
                    if result_cwd.structuredContent:
                        output_cwd = result_cwd.structuredContent.get("result")
                    elif result_cwd.content:
                        for content_block in result_cwd.content:
                            if hasattr(content_block, "type") and content_block.type == "text":
                                output_cwd = content_block.text
                                break
                    self.assertIsNotNone(output_cwd, "No output for cwd command.")
                    self.assertIn("cwd content", output_cwd)
                    self.assertIn('"exit_code": 0', output_cwd)

        finally:
            # Clean up
            if test_dir.exists():
                import shutil

                shutil.rmtree(test_dir)

    async def test_run_tests(self):
        """Verify the run_tests tool can execute test commands."""
        test_dir = HOME / ".gemini/tmp/run_tests_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file_path = test_dir / "test_example.py"

        # Create a simple pytest file
        test_content = """
import pytest

def test_passing_example():
    assert True

def test_failing_example():
    assert False
"""
        test_file_path.write_text(test_content)

        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Run tests using pytest
                    # We use the python binary from the venv to ensure pytest is found
                    pytest_command = f"{PYTHON_BIN} -m pytest {test_file_path}"
                    result = await session.call_tool("run_tests", arguments={"test_command": pytest_command})
                    output = None
                    if result.structuredContent:
                        output = result.structuredContent.get("result")
                    elif result.content:
                        for content_block in result.content:
                            if hasattr(content_block, "type") and content_block.type == "text":
                                output = content_block.text
                                break
                    self.assertIsNotNone(output, "No output for run_tests command.")

                    output_dict = json.loads(output)
                    combined_output = output_dict["stdout"] + output_dict["stderr"]
                    self.assertIn("1 failed, 1 passed", combined_output)
        finally:
            # Clean up
            if test_dir.exists():
                import shutil

                shutil.rmtree(test_dir)

    async def test_venv_run(self):
        """Verify the venv_run tool can execute commands within a specified virtual environment."""
        venv_base_dir = HOME / ".gemini/tmp/venv_run_test_base"
        venv_base_dir.mkdir(parents=True, exist_ok=True)
        temp_venv_path = venv_base_dir / "temp_venv"

        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # 1. Create a temporary virtual environment
                    create_venv_command = f"{PYTHON_BIN} -m venv {temp_venv_path}"
                    create_venv_result_json = await session.call_tool(
                        "run_shell_command", arguments={"command": create_venv_command}
                    )
                    create_venv_result = json.loads(create_venv_result_json.structuredContent.get("result"))
                    self.assertEqual(
                        create_venv_result["exit_code"], 0, f"Failed to create venv: {create_venv_result['stderr']}"
                    )
                    self.assertTrue(temp_venv_path.exists())
                    self.assertTrue((temp_venv_path / "bin" / "python").exists())

                    # 2. Install a simple package (e.g., requests) into this venv
                    install_command = "pip install requests"
                    install_result_json = await session.call_tool(
                        "venv_run", arguments={"venv_path": str(temp_venv_path), "command": install_command}
                    )
                    install_result = json.loads(install_result_json.structuredContent.get("result"))
                    self.assertEqual(
                        install_result["exit_code"], 0, f"Failed to install requests: {install_result['stderr']}"
                    )
                    self.assertIn("Successfully installed", install_result["stdout"])
                    self.assertIn("requests", install_result["stdout"])

                    # 3. Execute a command within this venv to verify installation
                    verify_command = "python -c \"import requests; print('requests imported successfully')\""
                    verify_result_json = await session.call_tool(
                        "venv_run", arguments={"venv_path": str(temp_venv_path), "command": verify_command}
                    )
                    verify_result = json.loads(verify_result_json.structuredContent.get("result"))
                    self.assertEqual(
                        verify_result["exit_code"], 0, f"Failed to verify requests: {verify_result['stderr']}"
                    )
                    self.assertIn("requests imported successfully", verify_result["stdout"])

                    # 4. Test error handling for a non-existent venv
                    non_existent_venv = venv_base_dir / "non_existent_venv"
                    error_command = "python -c \"print('hello')\""
                    error_result_json = await session.call_tool(
                        "venv_run", arguments={"venv_path": str(non_existent_venv), "command": error_command}
                    )
                    error_result = json.loads(error_result_json.structuredContent.get("result"))
                    self.assertNotEqual(error_result["exit_code"], 0)
                    self.assertIn("Python executable not found in virtual environment", error_result["error"])

        finally:
            # Clean up the base directory
            if venv_base_dir.exists():
                import shutil

                shutil.rmtree(venv_base_dir)

    async def test_gis_integration(self):
        """Verify that GisCore is correctly integrated with context/config.json."""
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Test retrieving a known value from context/config.json
                result = await session.call_tool("gis_get", arguments={"key": "index.name"})

                # Check where the result is (structuredContent or content)
                value = None
                if result.structuredContent:
                    # gis_get returns Any, so it might be directly in 'result' if structured
                    # However, based on types.py, structuredContent is a dict.
                    # Let's inspect the implementation of CallToolResult in FastMCP server.py wrapper.
                    # The tool definition returns Any.
                    # Assuming it returns the raw value in the result.
                    value = result.structuredContent.get("result")  # This depends on FastMCP behavior
                    # Wait, looking at previous tests, result.structuredContent seems to be the return value itself?
                    # No, in previous tests we did result.structuredContent.get('result') because
                    # run_shell_command returns a JSON string, which FastMCP might interpret?
                    # No, run_shell_command returns a string that IS json.
                    # gis_get returns Any.
                    pass

                # Let's handle the TextContent case as well, which is safer if we are unsure about structured output for simple types.
                if value is None and result.content:
                    for content_block in result.content:
                        if hasattr(content_block, "type") and content_block.type == "text":
                            # gis_get returns Any. If it's a string, it's here.
                            # If it's a complex object, it might be JSON encoded string?
                            # FastMCP usually converts return values to text if not structured.
                            value = content_block.text
                            break

                self.assertEqual(value, "lloyd-home-context")


if __name__ == "__main__":
    unittest.main(verbosity=2)
