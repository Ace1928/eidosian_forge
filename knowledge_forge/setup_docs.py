#!/usr/bin/env python3
import os
import datetime

def create_dir(path):
    os.makedirs(path, exist_ok=True)
    print(f"[DIR CREATED] {path}")

def create_file(path, content=""):
    # Ensure the parent directory exists
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[FILE CREATED] {path}")

def default_header(file_path):
    # Use the file name (without extension) as title
    base = os.path.basename(file_path)
    title = os.path.splitext(base)[0].replace("_", " ").capitalize()
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    header = f"---\ntitle: \"{title}\"\ndate: {date}\ndescription: \"Placeholder for {file_path}\"\n---\n\n"
    return header

def main():
    print("Welcome to the Doc_Forge Hugo Documentation Setup Script!")
    language = input("Enter your language folder name (e.g. python): ").strip()
    locale = input("Enter your locale code (e.g. en): ").strip()
    
    # List of directories to create (with <language> and <locale> replaced)
    directories = [
        "content/docs/manual/_common/project",
        "content/docs/manual/_common/glossary",
        "content/docs/manual/_common/standards",
        f"content/docs/manual/{language}/guides/quickstart",
        f"content/docs/manual/{language}/guides/intermediate",
        f"content/docs/manual/{language}/guides/advanced",
        f"content/docs/manual/{language}/api/modules",
        f"content/docs/manual/{language}/api/classes",
        f"content/docs/manual/{language}/api/functions",
        f"content/docs/manual/{language}/design/patterns",
        f"content/docs/manual/{language}/design/decisions",
        f"content/docs/manual/{language}/design/diagrams",
        f"content/docs/manual/{language}/examples/snippets",
        f"content/docs/manual/{language}/examples/tutorials",
        f"content/docs/manual/{language}/examples/projects",
        f"content/docs/manual/{language}/best_practices/style",
        f"content/docs/manual/{language}/best_practices/patterns",
        f"content/docs/manual/{language}/best_practices/antipatterns",
        f"content/docs/manual/{language}/troubleshooting/errors",
        f"content/docs/manual/{language}/troubleshooting/debugging",
        f"content/docs/manual/{language}/troubleshooting/solutions",
        f"content/docs/manual/{language}/security/vulnerabilities",
        f"content/docs/manual/{language}/security/best_practices",
        f"content/docs/manual/{language}/security/auditing",
        f"content/docs/manual/{language}/changelog",
        f"content/docs/manual/{language}/contributing",
        f"content/docs/manual/{language}/faq",
        "content/docs/auto/_common/metrics",
        "content/docs/auto/_common/coverage",
        f"content/docs/auto/{language}/api/modules",
        f"content/docs/auto/{language}/api/classes",
        f"content/docs/auto/{language}/api/functions",
        f"content/docs/auto/{language}/models/schemas",
        f"content/docs/auto/{language}/models/entities",
        f"content/docs/auto/{language}/models/migrations",
        f"content/docs/auto/{language}/functions/signatures",
        f"content/docs/auto/{language}/functions/parameters",
        f"content/docs/auto/{language}/functions/returns",
        f"content/docs/auto/{language}/error_handling/exceptions",
        f"content/docs/auto/{language}/error_handling/error_codes",
        f"content/docs/auto/{language}/error_handling/recovery",
        f"content/docs/auto/{language}/benchmarks/results",
        f"content/docs/auto/{language}/benchmarks/comparisons",
        f"content/docs/auto/{language}/benchmarks/trends",
        f"content/docs/auto/{language}/internal/private",
        f"content/docs/auto/{language}/internal/protected",
        f"content/docs/auto/{language}/internal/helpers",
        f"content/docs/auto/{language}/schemas/database",
        f"content/docs/auto/{language}/schemas/storage",
        f"content/docs/auto/{language}/schemas/transport",
        f"content/docs/auto/{language}/configuration/options",
        f"content/docs/auto/{language}/configuration/defaults",
        f"content/docs/auto/{language}/configuration/examples",
        "content/docs/ai/explanations/concepts",
        "content/docs/ai/explanations/algorithms",
        "content/docs/ai/explanations/patterns",
        "content/docs/ai/examples/use_cases",
        "content/docs/ai/examples/implementations",
        "content/docs/ai/examples/integrations",
        "content/docs/ai/troubleshooting/common_errors",
        "content/docs/ai/troubleshooting/debugging",
        "content/docs/ai/troubleshooting/optimizations",
        "content/docs/external/references/libraries",
        "content/docs/external/references/frameworks",
        "content/docs/external/references/tools",
        "content/docs/external/integrations/apis",
        "content/docs/external/integrations/services",
        "content/docs/external/integrations/platforms",
        "content/assets/images/screenshots",
        "content/assets/images/logos",
        "content/assets/images/icons",
        "content/assets/diagrams/architecture",
        "content/assets/diagrams/flows",
        "content/assets/diagrams/sequences",
        "content/assets/css/themes",
        "content/assets/css/components",
        "content/assets/js/interactions",
        "content/assets/js/visualizations",
        "content/assets/fonts/web",
        "content/assets/fonts/print",
        "content/assets/videos/tutorials",
        "content/assets/videos/demos",
        "content/assets/videos/webinars",
        f"content/i18n/{locale}/manual",
        f"content/i18n/{locale}/auto",
        "content/versions/v1.0.0/manual",
        "content/versions/v1.0.0/auto",
        "content/tools/generators/api",
        "content/tools/generators/diagrams",
        "content/tools/generators/examples",
        "content/tools/linters/markdown",
        "content/tools/linters/code",
        "content/tools/linters/links",
        "content/tools/validators/structure",
        "content/tools/validators/content",
        "content/tools/validators/accessibility",
        "content/tools/templates/pages",
        "content/tools/templates/sections",
        "content/tools/templates/components",
        "content/tests/link-checker",
        "content/tests/code-samples",
        "content/tests/accessibility",
        "content/tests/rendering",
        "content/config/sphinx",
        "content/config/mkdocs",
        "content/config/docfx",
        "content/config/docusaurus",
        "content/config/jsdoc",
        "content/hooks/pre-commit",
        "content/hooks/post-build",
        "content/hooks/deployment"
    ]
    
    for dir_path in directories:
        create_dir(dir_path)
    
    # List of files to create with their default header (substitute <language> and <locale> appropriately)
    files = [
        "content/docs/manual/_common/project/overview.md",
        "content/docs/manual/_common/project/architecture.md",
        "content/docs/manual/_common/project/roadmap.md",
        "content/docs/manual/_common/project/team.md",
        "content/docs/manual/_common/index.md",
        f"content/docs/manual/{language}/changelog/latest.md",
        f"content/docs/manual/{language}/changelog/archive.md",
        f"content/docs/manual/{language}/contributing/code.md",
        f"content/docs/manual/{language}/contributing/docs.md",
        f"content/docs/manual/{language}/contributing/reviews.md",
        f"content/docs/manual/{language}/faq/general.md",
        f"content/docs/manual/{language}/faq/technical.md",
        f"content/docs/manual/{language}/faq/troubleshooting.md",
        f"content/docs/manual/{language}/index.md",
        "content/docs/manual/index.md",
        "content/docs/auto/_common/index.md",
        f"content/docs/auto/{language}/index.md",
        "content/docs/auto/index.md",
        "content/docs/ai/index.md",
        "content/docs/external/index.md",
        "content/assets/css/custom.css",
        "content/assets/js/main.js",
        "content/assets/README.md",
        f"content/i18n/{locale}/index.md",
        "content/i18n/config.yaml",
        "content/i18n/README.md",
        "content/versions/v1.0.0/index.md",
        "content/versions/current.md",
        "content/versions/changelog.md",
        "content/versions/migration.md",
        "content/config/.docs-config.yaml",
        "content/index.md",
        "content/search.md",
        "content/sitemap.xml",
        "content/.readthedocs.yaml",
        "content/_redirects",
        "content/robots.txt",
        "content/CONTRIBUTING.md",
        "content/README.md"
    ]
    
    for file_path in files:
        header = default_header(file_path)
        create_file(file_path, header)
    
    print("\nUniversal documentation structure setup is complete!")

if __name__ == "__main__":
    main()
