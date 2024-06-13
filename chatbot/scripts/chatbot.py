import pdfplumber
import os
import nltk

try:
    import torch
    print("PyTorch version:", torch.__version__)
except ImportError as e:
    print("Error importing PyTorch:", e)

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except pdfplumber.PDFException as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    except IOError as e:
        print(f"Error opening PDF file: {e}")
        return None

def save_text_to_file(text, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
    except IOError as e:
        print(f"Error writing to file: {e}")
        
        
# Directory path

directory = "D://MajorProject/chatbot/knowledge_base"

# Iterate over files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".pdf"):
        # Path to the PDF file
        pdf_path = os.path.join(directory, filename)
        output_file_path = os.path.join(directory, f"{os.path.splitext(filename)[0]}_manifesto.txt")
        if os.path.exists(output_file_path):
            continue

        # Extract text from the PDF using pdfplumber
        manifesto_text = extract_text_from_pdf(pdf_path)

        # Save the extracted text to a text file
        save_text_to_file(manifesto_text, output_file_path)

party_keywords = {
    "bjp": ["bjp", "bharatiya janata party", "narendra", "modi"],
    "inc": ["congress", "indian national congress", "rahul", "gandhi"],
    "aap": ["aap", "aam aadmi party", "arvind", "kejriwal"],
    "sp": ["sp", "samajwadi party", "akhilesh", "yadav"],
    "bsp": ["bsp", "bahujan samaj party", "mayawati"],
    "aitc": ["trinamool", "aitc", "mamata", "bannerjee"],
    "cpim": ["cpim", "communist"],
    "dmk": ["dmk", "south"]
}

import re

def identify_party(user_input):
    for party, keywords in party_keywords.items():
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            if pattern.search(user_input):
                return party
    return None

def load_manifesto_by_party(party):
    manifesto_file_path = f"../knowledge_base/{party.lower()}_manifesto.txt"
    try:
        with open(manifesto_file_path, "r", encoding="utf-8") as file:
            manifesto_text = file.read()
        return manifesto_text
    except FileNotFoundError:
        print(f"Manifesto for {party} not found.")
        return ""  # Return empty string if manifesto not found
    except Exception as e:
        print(f"Error loading manifesto for {party}: {str(e)}")
        return ""

nltk.download('punkt')

def answer_query_based_on_manifesto(user_query, party_name):
    # Load manifesto text
    manifesto_text = load_manifesto_by_party(party_name)

    if manifesto_text == "Manifesto not found.":
        return "Manifesto for the given party not found."

    # Preprocess user query
    user_query = user_query.lower()  # Convert to lowercase for case-insensitive matching

    # Tokenize user query
    query_tokens = nltk.word_tokenize(user_query)

    # Tokenize manifesto text into sentences
    manifesto_sentences = nltk.sent_tokenize(manifesto_text)

    # Initialize list to store relevant sentences
    relevant_sentences = []

    # Calculate similarity scores between user query and each manifesto sentence
    similarity_scores = []
    for sentence in manifesto_sentences:
        # Tokenize manifesto sentence
        sentence_tokens = nltk.word_tokenize(sentence)

        # Calculate Jaccard similarity between query and sentence tokens
        similarity = len(set(query_tokens) & set(sentence_tokens)) / len(set(query_tokens) | set(sentence_tokens))
        similarity_scores.append(similarity)

        # If similarity score exceeds a threshold, consider the sentence relevant
        if similarity > 0.5:  # Adjust the threshold as needed
            relevant_sentences.append(sentence)

    # If no relevant sentences are found
    if not relevant_sentences:
        return "I couldn't find relevant information in the manifesto for the given query."

    # Concatenate relevant sentences to form the response
    response = " ".join(relevant_sentences)

    return response

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

import os

manifesto_directory = "../knowledge_base/"

party_manifestos = {}

for filename in os.listdir(manifesto_directory):
    if filename.endswith(".txt"):
        party_name = os.path.splitext(filename)[0].replace("_manifesto", "")
        with open(os.path.join(manifesto_directory, filename), "r", encoding='utf_8') as file:
            manifesto_text = file.read()
        party_manifestos[party_name] = manifesto_text

tokenizer = T5Tokenizer.from_pretrained("t5-small")

class PartyManifestoDataset(torch.utils.data.Dataset):
    def __init__(self, party_manifestos, tokenizer):
        self.party_manifestos = party_manifestos
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        party = list(self.party_manifestos.keys())[idx]
        manifesto_text = self.party_manifestos[party]
        inputs = self.tokenizer.encode_plus(
            manifesto_text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels = self.tokenizer.encode_plus(
            manifesto_text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return inputs, labels, party

    def __len__(self):
        return len(self.party_manifestos)

dataset = PartyManifestoDataset(party_manifestos, tokenizer)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

model = T5ForConditionalGeneration.from_pretrained("t5-small")
device = torch.device("cpu")
model.to(device)