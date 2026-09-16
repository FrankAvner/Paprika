#!/bin/bash

cd "$(dirname "$0")"

echo "Activating Paprika virtual environment..."
source venv/bin/activate

echo "Starting Paprika..."
python src/ui/main_window.py

echo
echo "Paprika has stopped."
read -p "Press Enter to close..."
