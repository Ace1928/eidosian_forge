#!/bin/bash

# Script to standardize all *_forge repositories to use master branch
echo "🔄 Starting repository standardization to master branch..."

# Helper function to handle repository setup
setup_repo() {
  local dir=$1
  echo -e "\n🔍 Processing repository: $dir"

  # Check if directory exists and is a git repository
  if [ ! -d "$dir/.git" ]; then
    echo "⚠️ Not a git repository or directory doesn't exist. Skipping."
    return
  fi

  # Go into the repository directory
  (
    cd "$dir" || { echo "❌ Failed to enter $dir. Skipping."; return; }

    # Save uncommitted changes if any
    if ! git diff --quiet || ! git diff --cached --quiet; then
      echo "💾 Saving uncommitted changes"
      git stash
      STASHED=1
    else
      STASHED=0
    fi

    # Create or set up the remote if needed
    REMOTE_EXISTS=$(git remote | grep -c "^origin$")
    REPO_NAME=$(basename "$dir")

    if [ "$REMOTE_EXISTS" -eq 0 ]; then
      echo "⚙️ Setting up remote origin"
      git remote add origin "https://github.com/Ace1928/${REPO_NAME}.git"
    else
      echo "⚙️ Ensuring remote URL is correct"
      git remote set-url origin "https://github.com/Ace1928/${REPO_NAME}.git"
    fi

    # Check if remote repository exists
    if ! git ls-remote --exit-code origin &>/dev/null; then
      echo "🆕 Remote repository doesn't exist yet. Will use local master only."
      REMOTE_EXISTS=0
    else
      REMOTE_EXISTS=1
      echo "✅ Remote repository exists."
      git fetch --all --quiet
    fi

    # Handle branch standardization

    # 1. Ensure master branch exists locally
    if ! git rev-parse --verify --quiet master &>/dev/null; then
      echo "🆕 Creating master branch"

      # Try to base it off main if it exists
      if git rev-parse --verify --quiet main &>/dev/null || git ls-remote --exit-code --heads origin main &>/dev/null; then
        echo "🔄 Using existing main branch as base"

        # Make sure we have local main
        if ! git rev-parse --verify --quiet main &>/dev/null; then
          git checkout -b main origin/main
        else
          git checkout main
        fi

        # Reset any conflicts if they exist
        git reset --hard

        # Create master based on main
        git checkout -b master
      else
        # No main exists, create master from current state
        echo "🆕 Creating new master branch from current state"
        git checkout -b master
      fi
    else
      # Master exists, just check it out
      echo "✅ Master branch exists locally"
      git checkout master

      # Reset any conflicts if they exist
      git reset --hard
    fi

    # 2. Handle main branch if it exists
    if git rev-parse --verify --quiet main &>/dev/null || git ls-remote --exit-code --heads origin main &>/dev/null; then
      echo "🔄 Main branch exists, syncing with master"

      # Ensure we have local main
      if ! git rev-parse --verify --quiet main &>/dev/null; then
        git checkout -b main origin/main
      else
        git checkout main
      fi

      # Make sure main is up-to-date with remote if possible
      if [ "$REMOTE_EXISTS" -eq 1 ]; then
        git pull origin main --ff-only || true
      fi

      # Reset any conflicts
      git reset --hard

      # Get commit hashes
      MAIN_COMMIT=$(git rev-parse HEAD)
      git checkout master
      MASTER_COMMIT=$(git rev-parse HEAD)

      if [ "$MAIN_COMMIT" = "$MASTER_COMMIT" ]; then
        echo "✅ Main and master are already identical"
      else
        echo "⚙️ Making master identical to main"
        git reset --hard main
      fi
    else
      echo "ℹ️ No main branch found, continuing with master only"
    fi

    # 3. Push master to remote if remote exists
    if [ "$REMOTE_EXISTS" -eq 1 ]; then
      echo "🔄 Pushing master to remote (with force if needed)"
      git push -f origin master
    else
      echo "ℹ️ Skipping push as remote doesn't exist"
    fi

    # 4. Set tracking
    git branch --set-upstream-to=origin/master master || true

    # 5. Restore stashed changes if any
    if [ "$STASHED" -eq 1 ]; then
      echo "🔄 Restoring uncommitted changes"
      git stash pop || true
    fi

    echo "✅ Completed standardization for $dir"
    echo "============================================="
  )
}

# Find all forge directories
FORGE_DIRS=$(find . -maxdepth 1 -type d -name "*_forge" -o -name "*_forge_repo")
echo "📂 Found forge directories: $FORGE_DIRS"

# Process each directory
for dir in $FORGE_DIRS; do
  setup_repo "$dir"
done

echo "🎉 All repositories have been standardized to use master branch!"
