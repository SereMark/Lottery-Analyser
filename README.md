# (optional) create a virtual env
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# install deps
pip install --upgrade pip
pip install streamlit pandas numpy scipy altair requests beautifulsoup4 lxml python-dateutil
