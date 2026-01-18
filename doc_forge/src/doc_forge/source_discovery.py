#!/usr/bin/env python3
# 🌀 Eidosian Source Discovery System
"""
Source Discovery - Finding Documentation Sources with Eidosian Precision
"""

import os
import re
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Optional, Any, Union, TypeVar, cast, Tuple

from .global_info import get_doc_structure
from .utils.paths import get_repo_root, get_docs_dir

PathStr = str
CategoryStr = str
SectionStr = str
TocItem = Dict[str, Any]
TocSection = Dict[str, Union[str, List[TocItem]]]
TocStructure = Dict[str, TocSection]
DocDict = Dict[CategoryStr, List['DocumentMetadata']]
T = TypeVar('T')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
)
logger = logging.getLogger("doc_forge.source_discovery")

class DocumentMetadata:
    def __init__(self, path: Path, title: str = "", category: str = "", section: str = "", priority: int = 50):
        self.path = path
        self.title = title or path.stem.replace("_", " ").title()
        self.category = category
        self.section = section
        self.priority = priority
        self.url = str(path.with_suffix(".html")).replace("\\", "/")
        self.references: Set[str] = set()
        self.is_index = path.stem.lower() == "index"
        self._extract_metadata()

    def _extract_metadata(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                content = f.read()
            if self.path.suffix == ".md":
                self._extract_markdown_metadata(content)
            elif self.path.suffix == ".rst":
                self._extract_rst_metadata(content)
            self._extract_common_references(content)
        except Exception as e:
            logger.warning(f"⚠️ Error extracting metadata from {self.path}: {e}")

    def _extract_markdown_metadata(self, content: str) -> None:
        title_match = re.search(r'^#\s+(.*?)$', content, re.MULTILINE)
        if title_match:
            self.title = title_match.group(1).strip()
        md_links = re.finditer(r'\[.*?\]\((.*?)\)', content)
        for match in md_links:
            link = match.group(1).strip()
            if not link.startswith(("http:", "https:", "#", "mailto:")):
                clean_link = re.sub(r'#.*$', '', link)
                clean_link = re.sub(r'\.(md|rst)$', '', clean_link)
                self.references.add(clean_link)

    def _extract_rst_metadata(self, content: str) -> None:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if i > 0 and i < len(lines) - 1:
                next_line = lines[i + 1]
                if re.match(r'^[=\-]+$', next_line) and len(line) > 0:
                    if len(line.strip()) == len(next_line.strip()):
                        self.title = line.strip()
                        break
        rst_links = re.finditer(r':doc:`(.*?)`', content)
        for match in rst_links:
            link = match.group(1).strip()
            self.references.add(link)
        rst_hyperlinks = re.finditer(r'`[^`]*?<(.*?)>`_', content)
        for match in rst_hyperlinks:
            link = match.group(1).strip()
            if not link.startswith(("http:", "https:", "#", "mailto:")):
                self.references.add(link)

    def _extract_common_references(self, content: str) -> None:
        include_patterns = [
            r'\.\. include:: (.*?)$',
            r'\{\% include "(.*?)" \%\}',
            r'\{\{ *include\("(.*?)"\) *\}\}',
        ]
        for pattern in include_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                include_path = match.group(1).strip()
                if not include_path.startswith(("http:", "https:", "#")):
                    self.references.add(include_path)

    def __repr__(self) -> str:
        return f"DocumentMetadata(title='{self.title}', path='{self.path}', category='{self.category}')"

class DocumentationDiscovery:
    def __init__(self, repo_root: Optional[Path] = None, docs_dir: Optional[Path] = None):
        self.repo_root = repo_root or get_repo_root()
        self.docs_dir = docs_dir or get_docs_dir()
        self.doc_structure = get_doc_structure(self.repo_root)
        self.documents: Dict[str, List[DocumentMetadata]] = defaultdict(list)
        self.orphaned_documents: List[Path] = []
        self.tracked_documents: Set[str] = set()
        self.doc_extensions = [".md", ".rst", ".txt"]
        self._file_content_cache: Dict[str, str] = {}

    def discover_all(self) -> Dict[str, List[DocumentMetadata]]:
        self._clear_discovery_state()
        self._discover_user_docs()
        self._discover_auto_docs()
        self._discover_ai_docs()
        self._identify_orphans()
        self._resolve_document_relations()
        return self.documents

    def _clear_discovery_state(self) -> None:
        self.documents.clear()
        self.orphaned_documents.clear()
        self.tracked_documents.clear()
        self._file_content_cache.clear()

    def _discover_user_docs(self) -> None:
        user_docs_dir = self.doc_structure.get("user_docs", self.docs_dir / "user_docs")
        if not user_docs_dir.exists():
            logger.warning(f"⚠️ User documentation directory not found at {user_docs_dir}")
            return
        try:
            for section in os.listdir(user_docs_dir):
                section_dir = user_docs_dir / section
                if not section_dir.is_dir():
                    continue
                for ext in self.doc_extensions:
                    for file_path in section_dir.glob(f"**/*{ext}"):
                        if any(p.startswith("_") for p in file_path.parts):
                            continue
                        doc = DocumentMetadata(
                            path=file_path,
                            category="user",
                            section=section,
                            priority=self._calculate_doc_priority(file_path, section, "user")
                        )
                        self.documents["user"].append(doc)
        except Exception as e:
            logger.error(f"🔥 Error discovering user documentation: {e}")
        logger.info(f"📚 Discovered {len(self.documents['user'])} user documentation files")

    def _discover_auto_docs(self) -> None:
        auto_docs_dir = self.doc_structure.get("auto_docs", self.docs_dir / "auto_docs")
        autodoc_dirs = [
            auto_docs_dir / "api",
            auto_docs_dir / "introspected",
            auto_docs_dir / "extracted",
            self.docs_dir / "autoapi",
            self.docs_dir / "apidoc",
            self.docs_dir / "reference",
        ]
        discovered_count = 0
        for section_dir in autodoc_dirs:
            if not section_dir.exists():
                continue
            section = section_dir.name
            for ext in self.doc_extensions:
                try:
                    for file_path in section_dir.glob(f"**/*{ext}"):
                        if any(p.startswith("_") for p in file_path.parts):
                            continue
                        doc = DocumentMetadata(
                            path=file_path,
                            category="auto",
                            section=section,
                            priority=self._calculate_doc_priority(file_path, section, "auto")
                        )
                        self.documents["auto"].append(doc)
                        discovered_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error processing auto docs in {section_dir}: {e}")
        logger.info(f"🤖 Discovered {discovered_count} auto-generated documentation files")

    def _discover_ai_docs(self) -> None:
        ai_docs_dir = self.doc_structure.get("ai_docs", self.docs_dir / "ai_docs")
        if not ai_docs_dir.exists():
            logger.debug(f"AI documentation directory not found at {ai_docs_dir}")
            return
        try:
            for section in os.listdir(ai_docs_dir):
                section_dir = ai_docs_dir / section
                if not section_dir.is_dir():
                    continue
                for ext in self.doc_extensions:
                    for file_path in section_dir.glob(f"**/*{ext}"):
                        if any(p.startswith("_") for p in file_path.parts):
                            continue
                        doc = DocumentMetadata(
                            path=file_path,
                            category="ai",
                            section=section,
                            priority=self._calculate_doc_priority(file_path, section, "ai")
                        )
                        self.documents["ai"].append(doc)
        except Exception as e:
            logger.error(f"🔥 Error discovering AI documentation: {e}")
        logger.info(f"🧠 Discovered {len(self.documents['ai'])} AI-generated documentation files")

    def _calculate_doc_priority(self, file_path: Path, section: str, category: str) -> int:
        base_priority = {"user": 40, "ai": 60, "auto": 80}.get(category, 50)
        if section in ["getting_started", "guides", "overview"]:
            base_priority -= 10
        if file_path.stem.lower() == "index":
            base_priority -= 20
        if file_path.stem.lower() == "readme":
            base_priority -= 15
        depth_penalty = len(file_path.parts) - len(self.docs_dir.parts) - 1
        base_priority += depth_penalty * 2
        return max(0, min(100, base_priority))

    def _identify_orphans(self) -> None:
        all_docs: List[Path] = []
        for ext in self.doc_extensions:
            all_docs.extend(list(self.docs_dir.glob(f"**/*{ext}")))
            all_docs.extend(self.docs_dir.glob(f"**/*{ext}"))
        structure_dirs = {
            str(path) for path in self.doc_structure.values()
            if hasattr(path, "exists") and path.exists()
        }
        ignored_dirs = [
            self.docs_dir / "_build",
            self.docs_dir / "_static",
            self.docs_dir / "_templates",
            self.docs_dir / "venv",
            self.docs_dir / ".venv",
        ]
        for file_path in all_docs:
            if any(p.startswith("_") for p in file_path.parts):
                continue
            if any(str(file_path).startswith(str(ignored)) for ignored in ignored_dirs):
                continue
            in_structure = False
            for dir_path in structure_dirs:
                if str(file_path).startswith(dir_path):
                    in_structure = True
                    break
            already_tracked = False
            doc_url = str(file_path.with_suffix(".html")).replace("\\", "/")
            if doc_url in self.tracked_documents:
                already_tracked = True
            if not in_structure and not already_tracked:
                self.orphaned_documents.append(file_path)
        logger.info(f"🏝️ Found {len(self.orphaned_documents)} orphaned documentation files")

    def _resolve_document_relations(self) -> None:
        doc_map: Dict[str, DocumentMetadata] = {}
        for _, docs in self.documents.items():
            for d in docs:
                doc_map[d.url] = d
        for _, docs_list in self.documents.items():
            for doc in docs_list:
                for ref in doc.references:
                    ref_url = f"{ref}.html"
                    if ref_url in doc_map:
                        pass
        logger.debug("📊 Document relations resolved with Eidosian precision")

    def generate_toc_structure(self) -> Dict[str, TocSection]:
        toc: Dict[str, TocSection] = {
            "getting_started": {"title": "Getting Started", "items": []},
            "user_guide": {"title": "User Guide", "items": []},
            "concepts": {"title": "Concepts", "items": []},
            "reference": {"title": "API Reference", "items": []},
            "examples": {"title": "Examples", "items": []},
            "advanced": {"title": "Advanced Topics", "items": []},
        }
        section_mapping = {
            "getting_started": "getting_started",
            "installation": "getting_started",
            "quickstart": "getting_started",
            "guides": "user_guide",
            "howto": "user_guide",
            "tutorials": "user_guide",
            "concepts": "concepts",
            "architecture": "concepts",
            "design": "concepts",
            "reference": "reference",
            "api": "reference",
            "examples": "examples",
            "demos": "examples",
            "advanced": "advanced",
            "internals": "advanced",
            "contributing": "advanced",
            "faq": "user_guide",
        }
        for _, docs_list in self.documents.items():
            for doc in docs_list:
                target_section = section_mapping.get(doc.section.lower(), "reference")
                if target_section not in toc:
                    toc[target_section] = {"title": target_section.title(), "items": []}
                items_list = toc[target_section]["items"]
                if isinstance(items_list, list):
                    items_list.append({
                        "title": doc.title,
                        "url": doc.url,
                        "priority": doc.priority,
                        "category": doc.category
                    })
                self.tracked_documents.add(doc.url)
        if self.orphaned_documents:
            self._place_orphaned_documents(toc)
        for sect in toc.values():
            if isinstance(sect["items"], list):
                typed_items = cast(List[TocItem], sect["items"]) # type: ignore
                sect["items"] = sorted(typed_items, key=lambda x: x.get("priority", 50))
        return toc

    def _place_orphaned_documents(self, toc: Dict[str, TocSection]) -> None:
        section_patterns: Dict[str, List[str]] = {
            "getting_started": ["installation", "quickstart", "setup", "introduction", "overview", "begin"],
            "user_guide": ["guide", "how to", "usage", "tutorial", "workflow", "manual", "instructions"],
            "concepts": ["concept", "architecture", "design", "principles", "theory", "philosophy", "model"],
            "reference": ["api", "reference", "class", "function", "method", "parameter", "attribute"],
            "examples": ["example", "sample", "demo", "showcase", "illustration", "walkthrough"],
            "advanced": ["advanced", "expert", "internals", "deep dive", "contribute", "extend", "customize"]
        }
        for orphan in self.orphaned_documents:
            if not orphan.is_file() or not orphan.exists():
                continue
            orphan_url = str(orphan.relative_to(self.docs_dir)).replace('\\', '/')
            orphan_url = re.sub(r'\.(md|rst|txt)$', '.html', orphan_url)
            if orphan_url in self.tracked_documents:
                continue
            best_section, title = self._analyze_orphan_content(orphan, section_patterns)
            items_list = toc[best_section]["items"]
            if isinstance(items_list, list):
                items_list.append({
                    "title": title,
                    "url": orphan_url,
                    "priority": 90,
                    "category": "orphan"  # Not unused - keeps value for documentation purposes
                })
            self.tracked_documents.add(orphan_url)

    def _analyze_orphan_content(self, orphan: Path, section_patterns: Dict[str, List[str]]) -> Tuple[str, str]:
        best_section = "reference"
        best_score = 0
        title = orphan.stem.replace("_", " ").title()
        filename_lower = orphan.stem.lower()
        for sect, patterns in section_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    if best_section == "reference" or best_score < 2:
                        best_section = sect
                        best_score = 2
        if best_score < 2:
            try:
                content_key = str(orphan)
                if content_key in self._file_content_cache:
                    content = self._file_content_cache[content_key].lower()
                else:
                    with open(orphan, "r", encoding="utf-8") as f:
                        content = f.read().lower()
                    self._file_content_cache[content_key] = content
                for sect, patterns in section_patterns.items():
                    section_score = sum(
                        content.count(pat) * (3 if pat in content[:500] else 1)
                        for pat in patterns
                    )
                    if section_score > best_score:
                        best_section = sect
                        best_score = section_score
            except Exception as e:
                logger.debug(f"Couldn't analyze orphan {orphan}: {e}")
        try:
            content_key = str(orphan)
            if content_key not in self._file_content_cache:
                with open(orphan, "r", encoding="utf-8") as f:
                    content_text = f.read()
                self._file_content_cache[content_key] = content_text
            else:
                content_text = self._file_content_cache[content_key]
            content_lines = content_text.split('\n')
            if orphan.suffix == ".md":
                for line in content_lines:
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
            elif orphan.suffix == ".rst":
                for i, line in enumerate(content_lines):
                    if i > 0 and i < len(content_lines) - 1:
                        next_line = content_lines[i + 1]
                        if (re.match(r'^[=\-]+$', next_line)
                                and line.strip()
                                and len(line.strip()) >= len(next_line.strip()) * 0.8):
                            title = line.strip()
                            break
        except Exception as e:
            logger.debug(f"Couldn't extract title from {orphan}: {e}")
        return best_section, title

    def find_all_sources(self) -> Dict[str, List[Path]]:
        all_sources: Dict[str, List[Path]] = {
            "user": [],
            "auto": [],
            "ai": [],
            "orphaned": []
        }
        user_docs_dir = self.doc_structure.get("user_docs", self.docs_dir / "user_docs")
        if user_docs_dir.exists():
            for ext in self.doc_extensions:
                all_sources["user"].extend(
                    p for p in user_docs_dir.glob(f"**/*{ext}")
                    if not any(part.startswith("_") for part in p.parts)
                )
        auto_docs_dir = self.doc_structure.get("auto_docs", self.docs_dir / "auto_docs")
        autoapi_dirs = [
            auto_docs_dir,
            self.docs_dir / "autoapi",
            self.docs_dir / "apidoc",
            self.docs_dir / "reference"
        ]
        for directory in autoapi_dirs:
            if directory.exists():
                for ext in self.doc_extensions:
                    all_sources["auto"].extend(
                        p for p in directory.glob(f"**/*{ext}")
                        if not any(part.startswith("_") for part in p.parts)
                    )
        ai_docs_dir = self.doc_structure.get("ai_docs", self.docs_dir / "ai_docs")
        if ai_docs_dir.exists():
            for ext in self.doc_extensions:
                all_sources["ai"].extend(
                    p for p in ai_docs_dir.glob(f"**/*{ext}")
                    if not any(part.startswith("_") for part in p.parts)
                )
        all_sources["orphaned"] = self.orphaned_documents
        return all_sources

    def resolve_references(self) -> Dict[str, Set[str]]:
        reference_map: Dict[str, Set[str]] = {}
        for _, docs_list in self.documents.items():  # Changed 'category' to '_' as it's unused
            for doc in docs_list:
                if doc.references:
                    reference_map[str(doc.path)] = doc.references
        return reference_map

def discover_documentation(docs_dir: Optional[Path] = None) -> Dict[str, List[DocumentMetadata]]:
    if docs_dir is None:
        docs_dir = get_docs_dir()
    logger.info(f"🔍 Discovering documentation in {docs_dir}")
    discovery = DocumentationDiscovery(docs_dir=docs_dir)
    documents = discovery.discover_all()
    logger.info(f"✅ Documentation discovery complete. Found {sum(len(docs) for docs in documents.values())} documents")
    return documents

def discover_code_structures(src_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    if src_dir is None:
        repo_root = get_repo_root()
        src_dir = repo_root / "src"
        if not src_dir.exists():
            for candidate in [repo_root, repo_root / "lib", repo_root / "source"]:
                if candidate.exists() and any(p.suffix == ".py" for p in candidate.glob("**/*.py")):
                    src_dir = candidate
                    break
    if not src_dir.exists():
        logger.warning(f"⚠️ Source directory not found at {src_dir}")
        return []
    logger.info(f"🔍 Discovering code structures in {src_dir}")
    discovered_items: List[Dict[str, Any]] = []
    python_files = list(src_dir.glob("**/*.py"))
    for file_path in python_files:
        try:
            if file_path.name == "__init__.py" or "test" in file_path.name.lower():
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            module_path = str(file_path.relative_to(src_dir.parent)).replace("\\", "/")
            module_name = module_path.replace("/", ".").replace(".py", "")
            discovered_items.append({
                "name": module_name,
                "type": "module",
                "file": str(file_path),
                "doc_ready": True
            })
            class_matches = re.finditer(r'class\s+([A-Za-z0-9_]+)(?:\(.*?\))?:', content)
            for match in class_matches:
                class_name = match.group(1)
                class_pos = match.start()
                docstring_match = re.search(r'"""(.*?)"""', content[class_pos:class_pos+500], re.DOTALL)
                has_docs = bool(docstring_match)
                discovered_items.append({
                    "name": class_name,
                    "type": "class",
                    "file": str(file_path),
                    "module": module_name,
                    "doc_ready": has_docs
                })
            func_matches = re.finditer(r'def\s+([A-Za-z0-9_]+)(?:\(.*?\))?:', content)
            for match in func_matches:
                func_name = match.group(1)
                if func_name.startswith("_") and not (func_name.startswith("__") and func_name.endswith("__")):
                    continue
                func_pos = match.start()
                docstring_match = re.search(r'"""(.*?)"""', content[func_pos:func_pos+500], re.DOTALL)
                has_docs = bool(docstring_match)
                discovered_items.append({
                    "name": func_name,
                    "type": "function",
                    "file": str(file_path),
                    "module": module_name,
                    "doc_ready": has_docs
                })
        except Exception as e:
            logger.warning(f"⚠️ Error processing {file_path}: {e}")
    logger.info(f"✅ Code structure discovery complete. Found {len(discovered_items)} items")
    return discovered_items

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Discover documentation files.")
    parser.add_argument("docs_dir", nargs="?", type=Path, help="Documentation directory")
    parser.add_argument("--output", "-o", type=str, help="Output file for discovered documents")
    parser.add_argument("--format", "-f", choices=["json", "yaml", "text"], default="text", help="Output format")
    args = parser.parse_args()
    discovered = discover_documentation(args.docs_dir)
    print(f"📚 Documentation Discovery Report:")
    for category, docs_list in discovered.items():
        print(f"  {category.title()}: {len(docs_list)} documents")
    if args.output:
        if args.format == "json":
            import json
            with open(args.output, "w") as f:
                json.dump({
                    cat: [{"title": d.title, "path": str(d.path)} for d in dl]
                    for cat, dl in discovered.items()
                }, f, indent=2)
        elif args.format == "yaml":
            try:
                import yaml
                with open(args.output, "w") as f:
                    yaml.dump({
                        cat: [{"title": d.title, "path": str(d.path)} for d in dl]
                        for cat, dl in discovered.items()
                    }, f)
            except ImportError:
                print("⚠️ PyYAML not installed. Using JSON format instead.")
                import json
                with open(args.output, "w") as f:
                    json.dump({
                        cat: [{"title": d.title, "path": str(d.path)} for d in dl]
                        for cat, dl in discovered.items()
                    }, f, indent=2)
        else:
            with open(args.output, "w") as f:
                for cat, dl in discovered.items():
                    f.write(f"=== {cat.title()} ===\n")
                    for d in dl:
                        f.write(f"{d.title} ({d.path})\n")
                    f.write("\n")
