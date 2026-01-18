#!/bin/bash
# 🌀 Eidosian Documentation Builder 
# Complete solution for building perfect documentation

# Self-aware error handling - Structure as Control
set -o pipefail
export TIMEFORMAT="Task completed in %3lR"

# Track execution time and success - Velocity as Intelligence
start_time=$(date +%s)
success_count=0
error_count=0
warning_count=0

# Utility functions for output formatting - Precision as Style
print_header() {
    echo ""
    echo "╔═════════════════════════════════════════════════════════════════╗"
    echo "║   🌀 $1 🌀 ║"
    echo "╚═════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_step() {
    echo "┌─────────────────────────────────────────────────────────────────┐"
    echo "│ $1"
    echo "└─────────────────────────────────────────────────────────────────┘"
}

print_success() {
    echo "✅ $1"
    ((success_count++))
}

print_error() {
    echo "❌ $1"
    ((error_count++))
}

print_warning() {
    echo "⚠️ $1"
    ((warning_count++))
}

execute_step() {
    local cmd="$1"
    local description="$2"
    local start_step=$(date +%s)
    
    print_step "$description"
    echo "$ $cmd"
    
    # Execute command and capture output
    output=$($cmd 2>&1)
    exit_code=$?
    
    # Calculate step duration
    local end_step=$(date +%s)
    local duration=$((end_step - start_step))
    
    if [ $exit_code -eq 0 ]; then
        print_success "$description completed in ${duration}s"
    else
        print_error "$description failed (${duration}s)"
        echo "Output:"
        echo "$output"
        
        # Continue despite errors unless fatal flag is set
        if [ "$3" == "fatal" ]; then
            echo "Fatal error encountered. Exiting."
            exit $exit_code
        fi
    fi
    
    return $exit_code
}

# Begin the documentation build process
print_header "EIDOSIAN DOCUMENTATION BUILD SYSTEM - PERFECT EXECUTION"

# Establish the documentation environment - Exhaustive but Concise
docs_dir="docs"
build_dir="$docs_dir/build/html"

# Ensure we have all required dependencies
execute_step "pip install -r $docs_dir/requirements.txt" "Installing required dependencies"

# Create the file structure for missing references
execute_step "chmod +x create_missing_files.sh" "Preparing file structure script" fatal
execute_step "./create_missing_files.sh" "Creating file structure for all cross-references"

# Fix cross-references in existing files
execute_step "python update_cross_references.py $docs_dir" "Fixing cross-references in existing files"

# Update toctrees to include all documents
if [ -f "update_toctrees.py" ]; then
    execute_step "python update_toctrees.py $docs_dir" "Updating TOC trees"
fi

# Add orphan directives to standalone files
if [ -f "update_orphan_directives.py" ]; then
    execute_step "python update_orphan_directives.py $docs_dir" "Adding orphan directives to standalone files"
fi

# Create build directory with proper permissions
execute_step "mkdir -p $build_dir" "Preparing build directory"

# Build the documentation
execute_step "python -m sphinx -b html $docs_dir $build_dir" "Building documentation" fatal

# Attempt stricter build to catch all warnings
print_step "Running strict validation build"
python -m sphinx -b html -W --keep-going $docs_dir ${build_dir}_strict > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Documentation passed strict validation (no warnings)"
else
    print_warning "Documentation contains warnings - see detailed log for info"
    # Generate warnings report
    python -m sphinx -b html -W --keep-going $docs_dir ${build_dir}_strict > /dev/null 2> warnings.log
    echo "Warnings saved to warnings.log"
fi

# Calculate full execution time
end_time=$(date +%s)
total_duration=$((end_time - start_time))

# Display completion summary
print_header "BUILD SUMMARY"
echo "📊 Operations: $((success_count + error_count + warning_count)) total"
echo "✅ Successful: $success_count"
echo "❌ Errors: $error_count"
echo "⚠️ Warnings: $warning_count"
echo "⏱️ Total execution time: ${total_duration}s"

# Success message with link to open the docs
if [ $error_count -eq 0 ]; then
    echo ""
    echo "📚 Documentation built successfully!"
    echo "📂 Your documentation is available at: $build_dir/index.html"
    
    # On Linux systems, offer to open the docs
    if [ "$(uname)" == "Linux" ]; then
        echo -n "🌐 Would you like to open the documentation now? [y/N] "
        read open_docs
        if [[ $open_docs == "y" || $open_docs == "Y" ]]; then
            xdg-open "$build_dir/index.html"
        fi
    fi
else
    echo ""
    echo "🔧 Documentation build completed with errors"
    echo "   Please review the output above and fix the issues"
fi
