#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing 7zip package..."
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    if apt-cache show 7zip >/dev/null 2>&1; then
        sudo apt-get install -y 7zip
    else
        sudo apt-get install -y p7zip-full
    fi
elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y p7zip p7zip-plugins
elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y p7zip p7zip-plugins
elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -Sy --noconfirm p7zip
elif command -v brew >/dev/null 2>&1; then
    brew install p7zip
else
    echo "Could not find a supported package manager for installing 7zip." >&2
    exit 1
fi

if ! command -v 7z >/dev/null 2>&1; then
    echo "7z command was not found after installation." >&2
    exit 1
fi

if [ ! -f "dataset.7z" ]; then
    echo "dataset.7z was not found in: $SCRIPT_DIR" >&2
    exit 1
fi

echo "Extracting dataset.7z into current directory..."
7z x "dataset.7z" -o"$SCRIPT_DIR" -y

echo "Installing Python packages..."
python -m pip install scikit-learn pandas matplotlib

echo "Setup complete."
