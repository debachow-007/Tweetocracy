from flask import Flask, render_template, request, jsonify
import pdfplumber
import re
import os

app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for searching the party manifesto
@app.route('/search', methods=['POST'])
def search():
    # Get the party and topic from the form data
    party = request.form['party'].lower()
    topic = request.form['topic']
    
    # Load the corresponding PDF file based on the selected party
    pdf_path = os.path.join('chatbot', 'knowledge_base', f'{party}.pdf')
    
    # Check if the PDF file exists
    if not os.path.exists(pdf_path):
        return jsonify({'error': 'Party manifesto not found'}), 404

    highlighted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                # Highlight the search results in the page text
                page_text_highlighted = re.sub(rf'({re.escape(topic)})', r'<span style="background-color: yellow;">\1</span>', page_text, flags=re.IGNORECASE)
                highlighted_text += page_text_highlighted + "\n"

    return jsonify({'pdf': highlighted_text})

if __name__ == '__main__':
    app.run(debug=True, port=4001)
