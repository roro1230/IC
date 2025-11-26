# create virtual environment
python3 -m venv .venv

# open a new terminal in VS Code then (if not auto-activated) activate manually
source .venv/bin/activate

# install deps
pip install -r requirements.txt

# run local
python3 app.py

The app will start on `http://localhost:5000`. Open this URL in your browser, or Ctrl+Click the link in the terminal.

## Usage

1. Open the web interface at `http://localhost:5000`
2. Upload a color image (JPG, PNG, etc.)
3. Select a model (CNN recommended for speed)
4. View results:
   - Original image
   - Grayscale version
   - Colorized output
