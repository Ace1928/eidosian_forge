#!/bin/bash

# Backup sources.list and sources.list.d
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo cp -r /etc/apt/sources.list.d /etc/apt/sources.list.d.bak

# Comment out problematic lines in sources.list
sudo sed -i '/archive.canonical.com\/ubuntu jammy multiverse/s/^/#/' /etc/apt/sources.list
sudo sed -i '/archive.canonical.com\/ubuntu jammy universe/s/^/#/' /etc/apt/sources.list

# Comment out problematic lines in sources.list.d
for file in /etc/apt/sources.list.d/*.list; do
    sudo sed -i '/archive.canonical.com\/ubuntu jammy multiverse/s/^/#/' "$file"
    sudo sed -i '/archive.canonical.com\/ubuntu jammy universe/s/^/#/' "$file"
done

# Update package list
sudo apt-get update