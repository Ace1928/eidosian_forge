"""
Configuration file generator for Eidosian repositories.

This module generates various configuration files needed for the repository,
ensuring consistent settings across the monorepo.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import importlib.util
import sys
import json

from ..core.files import write_file
from ..core.templates import render_template
from ..constants.content import (
    README_TEMPLATE,
    GITIGNORE_CONTENT,
    EDITORCONFIG_CONTENT,
    EIDOSIAN_CONFIG_TEMPLATE,
    CI_CONFIG_TEMPLATE,
    CONTRIBUTING_TEMPLATE,
    CODE_OF_CONDUCT_CONTENT,
    LICENSE_TEMPLATE,
    SECURITY_TEMPLATE,
    CHANGELOG_TEMPLATE
)


def _load_global_info() -> Dict[str, Any]:
    """
    Load global information from the global_info module.
    
    Gracefully falls back to default values if the module can't be loaded,
    ensuring consistent behavior in any environment.
    
    Returns:
        Dictionary with author information and dependencies
    """
    try:
        # Try to import global_info module if it's in the path
        global_info_path = Path("/home/lloyd/repos/global_info.py")
        if global_info_path.exists():
            spec = importlib.util.spec_from_file_location("global_info", global_info_path)
            if spec and spec.loader:
                global_info = importlib.util.module_from_spec(spec)
                sys.modules["global_info"] = global_info
                spec.loader.exec_module(global_info)
                logging.debug("Successfully loaded global_info from %s", global_info_path)
                return {
                    "author": global_info.AUTHOR_INFO,
                    "license": global_info.MIT_LICENSE,
                    "deps": {
                        "dev": global_info.DEV_DEPENDENCIES,
                        "docs": global_info.DOCS_DEPENDENCIES,
                        "core": global_info.CORE_DEPENDENCIES,
                    }
                }
    except Exception as e:
        logging.debug(f"Could not load global_info: {e}")
    
    # Fallback values - optimized for minimal duplication while maintaining functionality
    logging.debug("Using fallback values for global_info")
    return {
        "author": {
            "name": "Eidosian Team",
            "email": "team@example.com",
            "org": "Eidosian",
            "org_email": "org@example.com",
            "github_user": "eidosian",
        },
        "license": LICENSE_TEMPLATE,
        "deps": {
            "dev": ["pytest", "black", "flake8", "mypy"],
            "docs": ["sphinx", "sphinx-rtd-theme"],
            "core": ["requests"],
        }
    }


def create_eidosian_json(base_path: Path, repo_name: str) -> bool:
    """
    Create .eidosian.json metadata file for a repository.
    
    Args:
        base_path: Base repository directory
        repo_name: Name of the repository
        
    Returns:
        True if file was created, False otherwise
    """
    now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    eidosian_data = {
        "version": "1.0.0",
        "principles": {
            "contextualIntegrity": {
                "active": True,
                "description": "Every file has its perfect place in the structure"
            },
            "humorAsCognitiveLeverage": {
                "active": True,
                "description": "Clear, memorable interfaces and error messages"
            },
            "exhaustiveButConcise": {
                "active": True,
                "description": "Complete coverage without redundancy"
            },
            "flowLikeRiver": {
                "active": True,
                "description": "Seamless integration between components"
            },
            "hyperPersonalYetUniversallyApplicable": {
                "active": True,
                "description": "Works across any context, from solo dev to enterprise"
            },
            "recursiveRefinement": {
                "active": True,
                "description": "Built for continuous improvement and iteration"
            },
            "precisionAsStyle": {
                "active": True,
                "description": "Clean, functional design as aesthetic principle"
            },
            "velocityAsIntelligence": {
                "active": True, 
                "description": "Optimized for rapid development and deployment"
            },
            "structureAsControl": {
                "active": True,
                "description": "Architecture defines capabilities"
            },
            "selfAwarenessAsFoundation": {
                "active": True,
                "description": "System introspects and optimizes itself"
            }
        },
        "metadata": {
            "creator": "repo_forge",
            "created": now_iso,
            "lastModified": now_iso,
            "description": f"A universal {repo_name} adhering to all Eidosian principles",
            "languagesSupported": [
                "python", "javascript", "typescript", "go", "rust", "java", "c++"
            ],
            "version_history": [
                {
                    "version": "1.0.0",
                    "date": now_iso,
                    "description": "Initial creation"
                }
            ]
        },
        "self_analysis": {
            "last_reflection": None,
            "optimization_suggestions": [],
            "usage_patterns": {}
        }
    }
    
    # Write with pretty formatting
    return write_file(
        base_path / ".eidosian.json", 
        json.dumps(eidosian_data, indent=2),
        overwrite=False
    )


def create_configuration_files(
    base_path: Path,
    repo_name: str,
    languages: Optional[List[str]] = None,
    overwrite: bool = True
) -> Dict[str, Any]:
    """
    Create configuration files for the repository.
    
    Args:
        base_path: Base repository directory
        repo_name: Name of the repository
        languages: Programming languages to support
        
    Returns:
        Dictionary with creation results
    """
    if languages is None:
        languages = ["python", "nodejs", "go", "rust"]
    
    created_files = []
    global_info = _load_global_info()
    
    # Create README.md
    readme_vars = {
        "repo_name": repo_name.replace("_", " ").title(),
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "current_year": datetime.now().strftime("%Y"),
        "author_name": global_info["author"]["name"],
        "author_email": global_info["author"]["email"],
        "org_name": global_info["author"]["org"],
        "github_user": global_info["author"]["github_user"]
    }
    readme_content = render_template(README_TEMPLATE, readme_vars)
    write_file(base_path / "README.md", readme_content)
    created_files.append("README.md")

    # Create .gitignore
    write_file(base_path / ".gitignore", GITIGNORE_CONTENT)
    created_files.append(".gitignore")

    # Create .editorconfig
    write_file(base_path / ".editorconfig", EDITORCONFIG_CONTENT)
    created_files.append(".editorconfig")

    # Create Eidosian config
    eidosian_config = render_template(EIDOSIAN_CONFIG_TEMPLATE, {
        "repo_name": repo_name,
        "current_date": datetime.now().strftime('%Y-%m-%d'),
        "author_name": global_info["author"]["name"],
        "author_email": global_info["author"]["email"],
        "org_name": global_info["author"]["org"],
    })
    write_file(base_path / "eidosian_config.yml", eidosian_config)
    created_files.append("eidosian_config.yml")
    
    # Create CI config
    ci_path = base_path / ".github" / "workflows"
    ci_path.mkdir(parents=True, exist_ok=True)
    ci_config = render_template(CI_CONFIG_TEMPLATE, {"repo_name": repo_name})
    write_file(ci_path / "ci.yml", ci_config)
    created_files.append(".github/workflows/ci.yml")
    
    # Create CONTRIBUTING.md
    contributing = render_template(CONTRIBUTING_TEMPLATE, {
        "repo_name": repo_name.replace("_", " ").title(),
        "author_name": global_info["author"]["name"],
        "author_email": global_info["author"]["email"]
    })
    write_file(base_path / "CONTRIBUTING.md", contributing)
    created_files.append("CONTRIBUTING.md")
    
    # Create CODE_OF_CONDUCT.md
    write_file(base_path / "CODE_OF_CONDUCT.md", CODE_OF_CONDUCT_CONTENT)
    created_files.append("CODE_OF_CONDUCT.md")
    
    # Create LICENSE
    license_content = global_info["license"] if isinstance(global_info["license"], str) else render_template(
        LICENSE_TEMPLATE, 
        {"current_year": datetime.now().strftime("%Y")}
    )
    write_file(base_path / "LICENSE", license_content)
    created_files.append("LICENSE")
    
    # Create SECURITY.md
    security = render_template(SECURITY_TEMPLATE, {
        "repo_name": repo_name,
        "author_email": global_info["author"]["email"]
    })
    write_file(base_path / "SECURITY.md", security)
    created_files.append("SECURITY.md")
    
    # Create CHANGELOG.md
    changelog = render_template(CHANGELOG_TEMPLATE, {
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "repo_name": repo_name
    })
    write_file(base_path / "CHANGELOG.md", changelog)
    created_files.append("CHANGELOG.md")
    
    # Create pyproject.toml if Python is in languages
    if "python" in languages:
        try:
            # Try to use global_info's helper if it exists
            sys.path.append("/home/lloyd/repos")
            from global_info import generate_pyproject_toml
            pyproject_content = generate_pyproject_toml(
                project_name=repo_name,
                description=f"Eidosian {repo_name} repository"
            )
            write_file(base_path / "pyproject.toml", pyproject_content)
            created_files.append("pyproject.toml")
        except ImportError:
            logging.debug("Could not import generate_pyproject_toml, skipping pyproject.toml generation")
    
    # Additional Eidosian-specific files for universal compatibility
    create_eidosian_json(base_path, repo_name)
    
    logging.info(f"Created {len(created_files)} configuration files")
    return {
        "success": True,
        "created_files": created_files,
        "base_path": str(base_path)
    }