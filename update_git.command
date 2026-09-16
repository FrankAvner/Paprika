#!/bin/bash

cd "$(dirname "$0")" || exit 1

echo "========================================"
echo "Updating Paprika Git repository"
echo "========================================"

echo
echo "Current folder:"
pwd

echo
echo "Git status:"
git status

echo
echo "Adding changes..."
git add .

echo
echo "Creating commit..."
git commit -m "Update Paprika"

echo
echo "Pushing to GitHub..."
git push

echo
echo "========================================"
echo "Git update completed"
echo "========================================"

read -p "Press Enter to close..."
