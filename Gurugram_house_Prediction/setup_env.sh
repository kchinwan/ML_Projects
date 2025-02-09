#!/bin/bash

# Set environment name
ENV_NAME="ds_env"

echo "🚀 Setting up the virtual environment: $ENV_NAME"

# Step 1: Create Virtual Environment
python -m venv $ENV_NAME

# Step 2: Activate Virtual Environment
source $ENV_NAME/bin/activate  # Mac/Linux
# For Windows (Uncomment below line if running manually)
# $ENV_NAME\Scripts\activate

# Step 3: Upgrade pip
pip install --upgrade pip

# Step 4: Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "❌ requirements.txt not found! Please place it in the same directory."
    exit 1
fi

# Step 5: Install JupyterLab and Register Kernel
pip install jupyterlab ipykernel
python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"

# Step 6: Confirm Installation
echo "✅ Virtual Environment setup complete!"
echo "To activate, run: source $ENV_NAME/bin/activate (Mac/Linux) or $ENV_NAME\Scripts\activate (Windows)"
echo "To launch JupyterLab, run: jupyter lab"
