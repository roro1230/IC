# create virtual environment
python3 -m venv .venv

# open a new terminal in VS Code then (if not auto-activated) activate manually
source .venv/bin/activate

# install deps
pip install -r requirements.txt

# run local
python3 app.py
Ctrl + Click on local url
