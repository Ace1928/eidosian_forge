📂 docs/                              # Main documentation directory
│
├── 📂 manual/                        # Handwritten documentation
│   ├── 📂 _common/                   # Cross-language shared documentation
│   │   ├── 📂 project/               # Project-level documentation
│   │   │   ├── 📜 overview.md        # Project overview and goals
│   │   │   ├── 📜 architecture.md    # High-level architecture overview
│   │   │   ├── 📜 roadmap.md         # Development roadmap and future plans
│   │   │   └── 📜 team.md            # Team structure and contacts
│   │   ├── 📂 glossary/              # Universal terminology definitions
│   │   ├── 📂 standards/             # Project-wide standards and conventions
│   │   └── 📜 index.md               # Entry point for common documentation
│   │
│   ├── 📂 <language>/                # Language-specific documentation (python, cpp, rust, etc.)
│   │   ├── 📂 guides/                # High-level guides & tutorials
│   │   │   ├── 📂 quickstart/        # Getting started guides
│   │   │   ├── 📂 intermediate/      # Mid-level complexity guides
│   │   │   └── 📂 advanced/          # Advanced usage patterns
│   │   │
│   │   ├── 📂 api/                   # Manually written API documentation
│   │   │   ├── 📂 modules/           # Module/package level documentation
│   │   │   ├── 📂 classes/           # Class documentation
│   │   │   └── 📂 functions/         # Function documentation
│   │   │
│   │   ├── 📂 design/                # Architectural decisions & overviews
│   │   │   ├── 📂 patterns/          # Design patterns used
│   │   │   ├── 📂 decisions/         # Architecture decision records (ADRs)
│   │   │   └── 📂 diagrams/          # Component and interaction diagrams
│   │   │
│   │   ├── 📂 examples/              # Example code explanations
│   │   │   ├── 📂 snippets/          # Short code examples
│   │   │   ├── 📂 tutorials/         # Step-by-step tutorials
│   │   │   └── 📂 projects/          # Complete example projects
│   │   │
│   │   ├── 📂 best_practices/        # Coding conventions & best practices
│   │   │   ├── 📂 style/             # Style guides
│   │   │   ├── 📂 patterns/          # Recommended patterns
│   │   │   └── 📂 antipatterns/      # What to avoid
│   │   │
│   │   ├── 📂 troubleshooting/       # Common issues and resolutions
│   │   │   ├── 📂 errors/            # Error message explanations
│   │   │   ├── 📂 debugging/         # Debugging techniques
│   │   │   └── 📂 solutions/         # Solutions to common problems
│   │   │
│   │   ├── 📂 security/              # Security guidelines and considerations
│   │   │   ├── 📂 vulnerabilities/   # Common vulnerabilities
│   │   │   ├── 📂 best_practices/    # Security best practices
│   │   │   └── 📂 auditing/          # Security audit procedures
│   │   │
│   │   ├── 📂 changelog/             # Version history & release notes
│   │   │   ├── 📜 latest.md          # Latest version changes
│   │   │   └── 📜 archive.md         # Historical changes
│   │   │
│   │   ├── 📂 contributing/          # Contribution guidelines
│   │   │   ├── 📜 code.md            # Code contribution guidelines
│   │   │   ├── 📜 docs.md            # Documentation contribution guidelines
│   │   │   └── 📜 reviews.md         # Code review processes
│   │   │
│   │   ├── 📂 faq/                   # Frequently Asked Questions
│   │   │   ├── 📜 general.md         # General project FAQs
│   │   │   ├── 📜 technical.md       # Technical FAQs
│   │   │   └── 📜 troubleshooting.md # Troubleshooting FAQs
│   │   │
│   │   └── 📜 index.md               # Entry point for language-specific manual documentation
│   │
│   └── 📜 index.md                   # Entry point for all manual documentation
│
├── 📂 auto/                          # Auto-generated documentation
│   ├── 📂 _common/                   # Cross-language auto-generated docs
│   │   ├── 📂 metrics/               # Project-wide metrics and statistics
│   │   ├── 📂 coverage/              # Test and documentation coverage
│   │   └── 📜 index.md               # Entry point for common auto-docs
│   │
│   ├── 📂 <language>/                # Auto-generated docs per language
│   │   ├── 📂 api/                   # AutoAPI-generated documentation
│   │   │   ├── 📂 modules/           # Module reference
│   │   │   ├── 📂 classes/           # Class reference
│   │   │   └── 📂 functions/         # Function reference
│   │   │
│   │   ├── 📂 models/                # Data models documentation
│   │   │   ├── 📂 schemas/           # Data schemas
│   │   │   ├── 📂 entities/          # Entity relationship diagrams
│   │   │   └── 📂 migrations/        # Migration documentation
│   │   │
│   │   ├── 📂 functions/             # Function-level documentation
│   │   │   ├── 📂 signatures/        # Function signatures
│   │   │   ├── 📂 parameters/        # Parameter details
│   │   │   └── 📂 returns/           # Return value documentation
│   │   │
│   │   ├── 📂 error_handling/        # Exception handling documentation
│   │   │   ├── 📂 exceptions/        # Exception class documentation
│   │   │   ├── 📂 error_codes/       # Error code reference
│   │   │   └── 📂 recovery/          # Error recovery strategies
│   │   │
│   │   ├── 📂 benchmarks/            # Performance benchmarks
│   │   │   ├── 📂 results/           # Benchmark results
│   │   │   ├── 📂 comparisons/       # Performance comparisons
│   │   │   └── 📂 trends/            # Performance trends over time
│   │   │
│   │   ├── 📂 internal/              # Internal API docs
│   │   │   ├── 📂 private/           # Private method documentation
│   │   │   ├── 📂 protected/         # Protected method documentation
│   │   │   └── 📂 helpers/           # Helper function documentation
│   │   │
│   │   ├── 📂 schemas/               # Database or data structure schemas
│   │   │   ├── 📂 database/          # Database schemas
│   │   │   ├── 📂 storage/           # Storage schemas
│   │   │   └── 📂 transport/         # Data transport formats (JSON, protobuf)
│   │   │
│   │   ├── 📂 configuration/         # Auto-generated config files & references
│   │   │   ├── 📂 options/           # Configuration options
│   │   │   ├── 📂 defaults/          # Default configurations
│   │   │   └── 📂 examples/          # Example configurations
│   │   │
│   │   └── 📜 index.md               # Entry point for auto-generated documentation
│   │
│   └── 📜 index.md                   # Entry point for all auto-generated documentation
│
├── 📂 ai/                            # AI-generated documentation
│   ├── 📂 explanations/              # AI-generated concept explanations
│   │   ├── 📂 concepts/              # Key concept explanations
│   │   ├── 📂 algorithms/            # Algorithm explanations
│   │   └── 📂 patterns/              # Pattern explanations
│   │
│   ├── 📂 examples/                  # AI-generated code examples
│   │   ├── 📂 use_cases/             # Use case examples
│   │   ├── 📂 implementations/       # Implementation examples
│   │   └── 📂 integrations/          # Integration examples
│   │
│   ├── 📂 troubleshooting/           # AI-assisted troubleshooting guides
│   │   ├── 📂 common_errors/         # Common error solutions
│   │   ├── 📂 debugging/             # AI-assisted debugging techniques
│   │   └── 📂 optimizations/         # Performance optimization tips
│   │
│   └── 📜 index.md                   # Entry point for AI-generated documentation
│
├── 📂 external/                      # Third-party dependency documentation
│   ├── 📂 references/                # References to external documentation
│   │   ├── 📂 libraries/             # Library references
│   │   ├── 📂 frameworks/            # Framework references
│   │   └── 📂 tools/                 # Tool references
│   │
│   ├── 📂 integrations/              # Integration documentation
│   │   ├── 📂 apis/                  # External API integrations
│   │   ├── 📂 services/              # External service integrations
│   │   └── 📂 platforms/             # Platform integrations
│   │
│   └── 📜 index.md                   # Entry point for external documentation
│
├── 📂 assets/                        # Static files (images, diagrams, styles)
│   ├── 📂 images/                    # Screenshots, diagrams, illustrations
│   │   ├── 📂 screenshots/           # Application screenshots
│   │   ├── 📂 logos/                 # Logos and branding
│   │   └── 📂 icons/                 # Icon sets
│   │
│   ├── 📂 diagrams/                  # Architecture flowcharts, UML diagrams
│   │   ├── 📂 architecture/          # Architecture diagrams
│   │   ├── 📂 flows/                 # Flow diagrams
│   │   └── 📂 sequences/             # Sequence diagrams
│   │
│   ├── 📂 css/                       # Custom stylesheets
│   │   ├── 📂 themes/                # Documentation themes
│   │   ├── 📂 components/            # Component-specific styles
│   │   └── 📜 custom.css             # Main custom stylesheet
│   │
│   ├── 📂 js/                        # JavaScript for interactive docs
│   │   ├── 📂 interactions/          # Interactive components
│   │   ├── 📂 visualizations/        # Data visualizations
│   │   └── 📜 main.js                # Main JavaScript file
│   │
│   ├── 📂 fonts/                     # Custom fonts
│   │   ├── 📂 web/                   # Web fonts
│   │   └── 📂 print/                 # Print-optimized fonts
│   │
│   ├── 📂 videos/                    # Video tutorials and demos
│   │   ├── 📂 tutorials/             # Tutorial videos
│   │   ├── 📂 demos/                 # Demo videos
│   │   └── 📂 webinars/              # Recorded webinars
│   │
│   └── 📜 README.md                  # Overview of assets directory
│
├── 📂 i18n/                          # Internationalization
│   ├── 📂 <locale>/                  # Localized content (e.g., fr, es, zh)
│   │   ├── 📂 manual/                # Translated manual docs
│   │   ├── 📂 auto/                  # Translated auto-generated docs
│   │   └── 📜 index.md               # Localized entry point
│   │
│   ├── 📜 config.yaml                # i18n configuration
│   └── 📜 README.md                  # i18n documentation
│
├── 📂 versions/                      # Version archives
│   ├── 📂 v1.0.0/                    # Documentation for version 1.0.0
│   │   ├── 📂 manual/                # Manual docs for v1.0.0
│   │   ├── 📂 auto/                  # Auto-generated docs for v1.0.0
│   │   └── 📜 index.md               # Version-specific entry point
│   │
│   ├── 📜 current.md                 # Current version reference
│   ├── 📜 changelog.md               # Version changelog
│   └── 📜 migration.md               # Migration guides between versions
│
├── 📂 tools/                         # Documentation tooling
│   ├── 📂 generators/                # Doc generation scripts
│   │   ├── 📂 api/                   # API doc generators
│   │   ├── 📂 diagrams/              # Diagram generators
│   │   └── 📂 examples/              # Example code generators
│   │
│   ├── 📂 linters/                   # Documentation linting tools
│   │   ├── 📂 markdown/              # Markdown linters
│   │   ├── 📂 code/                  # Code example linters
│   │   └── 📂 links/                 # Link validators
│   │
│   ├── 📂 validators/                # Doc validation utilities
│   │   ├── 📂 structure/             # Structure validators
│   │   ├── 📂 content/               # Content validators
│   │   └── 📂 accessibility/         # Accessibility validators
│   │
│   └── 📂 templates/                 # Documentation templates
│       ├── 📂 pages/                 # Page templates
│       ├── 📂 sections/              # Section templates
│       └── 📂 components/            # Component templates
│
├── 📂 tests/                         # Documentation testing
│   ├── 📂 link-checker/              # Tests for broken links
│   ├── 📂 code-samples/              # Tests for code examples
│   ├── 📂 accessibility/             # Accessibility validation
│   └── 📂 rendering/                 # Rendering tests
│
├── 📂 config/                        # Configuration files
│   ├── 📂 sphinx/                    # Sphinx configuration
│   ├── 📂 mkdocs/                    # MkDocs configuration
│   ├── 📂 docfx/                     # DocFX configuration
│   ├── 📂 docusaurus/                # Docusaurus configuration
│   ├── 📂 jsdoc/                     # JSDoc configuration
│   └── 📜 .docs-config.yaml          # Universal docs configuration
│
├── 📂 hooks/                         # CI/CD integration
│   ├── 📂 pre-commit/                # Pre-commit hooks
│   ├── 📂 post-build/                # Post-build processes
│   └── 📂 deployment/                # Deployment scripts
│
├── 📜 index.md                       # Main documentation entry point
├── 📜 search.md                      # Search functionality for documentation
├── 📜 sitemap.xml                    # Documentation sitemap
├── 📜 .readthedocs.yaml              # ReadTheDocs config
├── 📜 _redirects                     # URL redirects for documentation
├── 📜 robots.txt                     # Search engine directives
├── 📜 CONTRIBUTING.md                # Documentation contribution guidelines
└── 📜 README.md                      # Documentation about documentation
