import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Path to your notebook
NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), 'scripts/chatbot.ipynb')

def run_notebook(notebook_path, user_query):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    # Inject parameters into the notebook
    param_cell = nbformat.v4.new_code_cell(f"user_query = '{user_query}'")
    nb.cells.insert(0, param_cell)
    
    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
    except CellExecutionError as e:
        return str(e)
    
    # Find the output of the cell containing chat_with_party_manifestos function
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if 'chat_with_party_manifestos' in cell.source:
                return cell.outputs[0]['text'].strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get('query', '')

    # Run the notebook with the user's query
    response = run_notebook(NOTEBOOK_PATH, user_query)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
