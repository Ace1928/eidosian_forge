"""
Project structure generator for language-specific projects.

This module creates project scaffolds for various programming languages
within the Eidosian monorepo structure.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging

from ..core.files import write_file
from ..constants.content import (
    PROJECT_READMES,
    PROJECT_CONFIG_FILES,
    PROJECT_MAIN_FILES
)


def create_project_scaffold_single(
    base_path: Path, 
    language: str, 
    overwrite: bool = True
) -> Dict[str, Any]:
    """
    Create a project scaffold for a single language.
    
    Args:
        base_path: Base repository directory
        language: Programming language for the project
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with creation results for that language
    """
    project_path = base_path / "projects" / f"{language}_project"
    
    # Make sure the project directory exists
    project_path.mkdir(exist_ok=True, parents=True)
    
    # Create basic structure if it doesn't exist
    src_path = project_path / "src"
    tests_path = project_path / "tests"
    src_path.mkdir(exist_ok=True)
    tests_path.mkdir(exist_ok=True)
    
    created_files = []
    
    # Create README
    readme_content = PROJECT_READMES.get(language, PROJECT_READMES["default"])
    readme_content = readme_content.replace("$project_name", f"{language}_project")
    write_file(project_path / "README.md", readme_content, overwrite)
    created_files.append(f"projects/{language}_project/README.md")
    
    # Create configuration files
    for filename, content in PROJECT_CONFIG_FILES.get(language, {}).items():
        file_path = project_path / filename
        write_file(file_path, content, overwrite)
        created_files.append(f"projects/{language}_project/{filename}")
    
    # Create main files
    main_path = src_path / f"{language}_project" if language == "python" else src_path
    main_path.mkdir(exist_ok=True, parents=True)
    
    for filename, content in PROJECT_MAIN_FILES.get(language, {}).items():
        file_path = main_path / filename
        write_file(file_path, content, overwrite)
        created_files.append(f"projects/{language}_project/src/{f'{language}_project/' if language == 'python' else ''}{filename}")
    
    # Create test files
    if language == "python":
        test_init = "# Test package\n"
        write_file(tests_path / "__init__.py", test_init, overwrite)
        created_files.append(f"projects/{language}_project/tests/__init__.py")
        
        test_content = """import unittest

class TestExample(unittest.TestCase):
    def test_example(self):
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
"""
        write_file(tests_path / "test_example.py", test_content)
        created_files.append(f"projects/{language}_project/tests/test_example.py")
        
    elif language == "nodejs":
        test_content = """const assert = require('assert');

describe('Example Test', function() {
  it('should pass', function() {
    assert.strictEqual(1, 1);
  });
});
"""
        write_file(tests_path / "example.test.js", test_content)
        created_files.append(f"projects/{language}_project/tests/example.test.js")
    
    elif language == "go":
        test_content = """package main

import "testing"

func TestExample(t *testing.T) {
    if 1 != 1 {
        t.Errorf("Expected 1 to equal 1")
    }
}
"""
        write_file(tests_path / "example_test.go", test_content)
        created_files.append(f"projects/{language}_project/tests/example_test.go")
    
    elif language == "rust":
        test_content = """#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
"""
        write_file(src_path / "lib.rs", test_content)
        created_files.append(f"projects/{language}_project/src/lib.rs")
    
    logging.info(f"Created {language} project scaffold with {len(created_files)} files")
    return {
        "success": True,
        "language": language,
        "created_files": created_files,
        "project_path": str(project_path),
        "count": int(len(created_files))  # Force integer
    }


def create_project_scaffold(
    base_path: Path, 
    languages: Union[str, List[str], Dict], 
    overwrite: bool = True
) -> Dict[str, Any]:
    """
    Create project scaffolds for one or more languages.
    
    Args:
        base_path: Base repository directory
        languages: Programming language(s) for the project.
                  Can be a string, list of strings, or dict with language keys
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with creation results. For multiple languages, includes
        results for each language and aggregated counts.
    """
    # Normalize input to handle different types
    if isinstance(languages, str):
        return create_project_scaffold_single(base_path, languages, overwrite)
    
    elif isinstance(languages, list):
        all_results = {
            "success": True,
            "languages": languages.copy(),
            "results": {},
            "count": 0
        }
        
        for lang in languages:
            if not isinstance(lang, str):
                raise TypeError(f"Language must be a string, got {type(lang)}")
            result = create_project_scaffold_single(base_path, lang, overwrite)
            all_results["results"][lang] = result
            all_results["count"] += int(result.get("count", 0))
            
            # If any language fails, mark the overall operation as failed
            if not result.get("success", False):
                all_results["success"] = False
        
        return all_results
    
    elif isinstance(languages, dict):
        # Extract language keys from the dictionary
        return create_project_scaffold(base_path, list(languages.keys()), overwrite)
    
    else:
        raise TypeError(f"Languages must be a string, list, or dict, got {type(languages)}")