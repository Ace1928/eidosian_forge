import os
import pytest
from pathlib import Path
from unittest import mock
from collections import defaultdict
from typing import Dict, List, Set, Any
from ..source_discovery import DocumentationDiscovery

#!/usr/bin/env python3
# Test module for source_discovery.py with Eidosian precision


# Import the module to test


class TestFindAllSources:
    """Test suite for the find_all_sources method with Eidosian thoroughness."""

    @pytest.fixture
    def mock_discovery(self):
        """Create a mock DocumentationDiscovery instance with controlled paths."""
        with mock.patch('doc_forge.source_discovery.get_repo_root') as mock_repo_root, \
             mock.patch('doc_forge.source_discovery.get_docs_dir') as mock_docs_dir, \
             mock.patch('doc_forge.source_discovery.get_doc_structure') as mock_structure:
            
            # Setup mock paths with perfect structural clarity
            mock_repo_root.return_value = Path("/mock/repo")
            mock_docs_dir.return_value = Path("/mock/repo/docs")
            
            # Create a mock structure with Eidosian precision
            mock_structure.return_value = {
                "user_docs": Path("/mock/repo/docs/user_docs"),
                "auto_docs": Path("/mock/repo/docs/auto_docs"),
                "ai_docs": Path("/mock/repo/docs/ai_docs")
            }
            
            discovery = DocumentationDiscovery()
            
            # Setup orphaned documents for testing
            discovery.orphaned_documents = [
                Path("/mock/repo/docs/orphaned1.md"),
                Path("/mock/repo/docs/orphaned2.rst")
            ]
            
            # Setup doc extensions for testing
            discovery.doc_extensions = [".md", ".rst", ".txt"]
            
            yield discovery

    def test_find_all_sources_empty_directories(self, mock_discovery):
        """Test finding sources when directories are empty with perfect precision."""
        # Setup mock exists and glob methods for empty directories
        with mock.patch.object(Path, 'exists') as mock_exists, \
             mock.patch.object(Path, 'glob') as mock_glob:
             
            mock_exists.return_value = False
            mock_glob.return_value = []
            
            # Execute the method with Eidosian clarity
            result = mock_discovery.find_all_sources()
            
            # Verify results with perfect structural awareness
            assert isinstance(result, dict)
            assert set(result.keys()) == {"user", "auto", "ai", "orphaned"}
            assert result["user"] == []
            assert result["auto"] == []
            assert result["ai"] == []
            assert len(result["orphaned"]) == 2
            assert all(isinstance(item, Path) for item in result["orphaned"])

    def test_find_all_sources_with_content(self, mock_discovery):
        """Test finding sources with content in all directories with Eidosian comprehension."""
        # Mock paths and existence checks
        with mock.patch.object(Path, 'exists') as mock_exists, \
             mock.patch.object(Path, 'glob') as mock_glob:
             
            # All directories exist
            mock_exists.return_value = True
            
            # Define mock returns for different glob patterns with perfect precision
            def mock_glob_side_effect(pattern):
                if "user_docs" in str(pattern):
                    return [
                        Path("/mock/repo/docs/user_docs/guide1.md"),
                        Path("/mock/repo/docs/user_docs/guide2.rst"),
                        Path("/mock/repo/docs/user_docs/_private/hidden.md")  # Should be excluded
                    ]
                elif "auto_docs" in str(pattern) or "autoapi" in str(pattern) or "apidoc" in str(pattern) or "reference" in str(pattern):
                    return [
                        Path("/mock/repo/docs/auto_docs/api/module.md"),
                        Path("/mock/repo/docs/reference/class.rst")
                    ]
                elif "ai_docs" in str(pattern):
                    return [
                        Path("/mock/repo/docs/ai_docs/concept1.md"),
                        Path("/mock/repo/docs/ai_docs/_drafts/draft.md")  # Should be excluded
                    ]
                return []
            
            mock_glob.side_effect = mock_glob_side_effect
            
            # Create mock parts property for Path objects to check _private directories
            with mock.patch.object(Path, 'parts', create=True, new_callable=mock.PropertyMock) as mock_parts:
                # Define parts based on path for _private directory detection
                def get_parts(path):
                    str_path = str(path)
                    if "_private" in str_path or "_drafts" in str_path:
                        return ["mock", "repo", "docs", "_private", "file.md"]
                    else:
                        return ["mock", "repo", "docs", "regular", "file.md"]
                
                # Configure the mock to return different parts based on the path
                mock_parts.side_effect = lambda: get_parts(mock_parts._mock_self)
                
                # Execute the method with Eidosian precision
                result = mock_discovery.find_all_sources()
                
                # Verify results with perfect structural awareness
                assert isinstance(result, dict)
                assert len(result["user"]) == 2  # 2 user docs (excluding _private)
                assert len(result["auto"]) == 2  # 2 auto docs
                assert len(result["ai"]) == 1    # 1 AI doc (excluding _drafts)
                assert len(result["orphaned"]) == 2  # 2 orphaned docs
                
                # Verify specific paths with Eidosian thoroughness
                user_paths = [str(p) for p in result["user"]]
                assert "/mock/repo/docs/user_docs/guide1.md" in user_paths
                assert "/mock/repo/docs/user_docs/guide2.rst" in user_paths
                
                auto_paths = [str(p) for p in result["auto"]]
                assert "/mock/repo/docs/auto_docs/api/module.md" in auto_paths
                assert "/mock/repo/docs/reference/class.rst" in auto_paths
                
                ai_paths = [str(p) for p in result["ai"]]
                assert "/mock/repo/docs/ai_docs/concept1.md" in ai_paths
                
                # Verify that _private and _drafts were excluded with perfect filtering
                all_paths = user_paths + auto_paths + ai_paths
                assert not any("_private" in p for p in all_paths)
                assert not any("_drafts" in p for p in all_paths)

    def test_find_all_sources_only_user_docs(self, mock_discovery):
        """Test finding sources when only user docs directory exists with precision."""
        # Mock paths and existence checks
        with mock.patch.object(Path, 'exists') as mock_exists, \
             mock.patch.object(Path, 'glob') as mock_glob:
             
            # Define mock exists to return True only for user_docs
            def mock_exists_side_effect(path):
                return "user_docs" in str(path)
                
            mock_exists.side_effect = mock_exists_side_effect
            
            # Define glob results for user_docs only
            def mock_glob_side_effect(pattern):
                if "user_docs" in str(pattern):
                    return [
                        Path("/mock/repo/docs/user_docs/guide1.md"),
                        Path("/mock/repo/docs/user_docs/guide2.rst")
                    ]
                return []
                
            mock_glob.side_effect = mock_glob_side_effect
            
            # Mock parts check for _private directories
            with mock.patch.object(Path, 'parts', create=True, new_callable=mock.PropertyMock) as mock_parts:
                mock_parts.return_value = ["mock", "repo", "docs", "regular", "file.md"]
                
                # Execute and verify with Eidosian precision
                result = mock_discovery.find_all_sources()
                
                assert len(result["user"]) == 2
                assert len(result["auto"]) == 0
                assert len(result["ai"]) == 0
                assert len(result["orphaned"]) == 2

    def test_find_all_sources_integration(self, tmp_path):
        """Integration test with real filesystem access for source discovery with Eidosian comprehension."""
        # Create a temporary directory structure
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        
        # Create user docs
        user_docs = docs_dir / "user_docs"
        user_docs.mkdir()
        (user_docs / "guide.md").write_text("# User Guide")
        (user_docs / "_private").mkdir()
        (user_docs / "_private" / "draft.md").write_text("# Draft")
        
        # Create auto docs
        auto_docs = docs_dir / "auto_docs"
        auto_docs.mkdir()
        (auto_docs / "api.rst").write_text("API Reference\n============")
        
        # Create ai docs
        ai_docs = docs_dir / "ai_docs"
        ai_docs.mkdir()
        (ai_docs / "concept.md").write_text("# AI Concept")
        
        # Create orphaned doc
        (docs_dir / "orphaned.md").write_text("# Orphaned")
        
        # Create discovery with actual paths
        with mock.patch('doc_forge.source_discovery.get_doc_structure') as mock_structure:
            mock_structure.return_value = {
                "user_docs": user_docs,
                "auto_docs": auto_docs,
                "ai_docs": ai_docs
            }
            
            discovery = DocumentationDiscovery(docs_dir=docs_dir)
            discovery.orphaned_documents = [docs_dir / "orphaned.md"]
            
            # Execute with Eidosian precision
            result = discovery.find_all_sources()
            
            # Verify with perfect structural awareness
            assert len(result["user"]) == 1  # Excluding _private
            assert len(result["auto"]) == 1
            assert len(result["ai"]) == 1
            assert len(result["orphaned"]) == 1
            
            # Verify specific files
            assert str(user_docs / "guide.md") in [str(p) for p in result["user"]]
            assert str(auto_docs / "api.rst") in [str(p) for p in result["auto"]]
            assert str(ai_docs / "concept.md") in [str(p) for p in result["ai"]]
            assert str(docs_dir / "orphaned.md") in [str(p) for p in result["orphaned"]]
            
            # Verify private file exclusion with Eidosian filtering
            assert str(user_docs / "_private" / "draft.md") not in [str(p) for p in result["user"]]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])