#!/bin/bash
set -e
cd /data/data/com.termux/files/home/eidosian_forge

echo "Stashing changes..."
git stash

echo "Pulling from remote..."
git pull origin main --rebase

echo "Popping stash..."
git stash pop || echo "No stash to pop or merge conflict occurred."
