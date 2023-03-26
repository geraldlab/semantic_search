Rem ******************************
Rem This script prepares a folder for python project with 3 folders(data, pages, model), setup python virtual environment (.venv), and install ipykernel libray
Rem script should be run in fi
cd semantic_search
mkdir model
mkdir log
python -m venv .venvsemantic_search
cd .venvsemantic_search\Scripts
activate
pip install ipykernel
ipython kernel install --user --name=semantic_search