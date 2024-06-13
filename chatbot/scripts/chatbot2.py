import os

def read_manifesto_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        manifesto_text = file.read()
    return manifesto_text

def preprocess_text(text):
    # Perform any necessary preprocessing steps, such as lowercasing, removing punctuation, etc.
    # For simplicity, let's just lowercase the text for now.
    preprocessed_text = text.lower()
    return preprocessed_text

def load_manifestos(directory):
    manifestos = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            party_name = filename.split('.')[0]
            file_path = os.path.join(directory, filename)
            manifesto_text = read_manifesto_txt(file_path)
            preprocessed_text = preprocess_text(manifesto_text)
            manifestos[party_name] = preprocessed_text
    return manifestos

# Directory containing the TXT files of party manifestos
manifesto_directory = 'chatbot/knowledge_base'

# Load and preprocess the manifesto texts
manifestos = load_manifestos(manifesto_directory)

import re

# Define patterns for party names and topics
party_patterns = {
    "bjp": ["bjp", "bharatiya janata party", "narendra", "modi"],
    "inc": ["congress", "indian national congress", "rahul", "gandhi"],
    "aap": ["aap", "aam aadmi party", "arvind", "kejriwal"],
    "sp": ["sp", "samajwadi party", "akhilesh", "yadav"],
    "bsp": ["bsp", "bahujan samaj party", "mayawati"],
    "aitc": ["trinamool", "aitc", "mamata", "bannerjee"],
    "cpim": ["cpim", "communist"],
    "dmk": ["dmk", "south"]
}

topic_patterns = {
    "women": ["women", "gender", "genders", "gender equality"],
    "education": ["education", "schools", "universities"],
}

def perform_ner(query):
    entities = []
    
    # Match party names in the query
    for party_name, patterns in party_patterns.items():
        for pattern in patterns:
            if re.search(r'\b' + re.escape(pattern) + r'\b', query, re.IGNORECASE):
                entities.append((party_name, "PARTY"))
                break  # Move to the next party name
    
    # Match topics in the query
    for topic, patterns in topic_patterns.items():
        for pattern in patterns:
            if re.search(r'\b' + re.escape(pattern) + r'\b', query, re.IGNORECASE):
                entities.append((topic, "TOPIC"))
                break  # Move to the next topic
    
    return entities

# Example user query
user_query = "What are BJP's plans across genders?"

# Perform named entity recognition on the user query
ner_results = perform_ner(user_query)

# Print the identified entities and their labels
print("Named Entities:")
for entity, label in ner_results:
    print(f"Entity: {entity}, Label: {label}")

def map_query_to_manifesto(query, manifestos):
    mapped_sections = []
    
    # Perform named entity recognition on the user query
    entities = perform_ner(query)
    
    # Extract party name and topic from the identified entities
    party_name = None
    topic = None
    for entity, label in entities:
        if label == "PARTY":
            party_name = entity
        elif label == "TOPIC":
            topic = entity
    
    # If both party name and topic are identified, search for relevant sections
    if party_name and topic:
        # Search for the party manifesto based on the identified party name
        if party_name in manifestos:
            manifesto_text = manifestos[party_name]
            
            # Search for sections related to the identified topic
            sections = re.findall(rf'\b{re.escape(topic)}\b.*?(?=\b[A-Z]|$)', manifesto_text, re.IGNORECASE)
            mapped_sections.extend(sections)
    
    return mapped_sections

# Example user query
user_query = "What are BJP's plans for women?"

# Map the user query to relevant sections in the party manifestos
mapped_sections = map_query_to_manifesto(user_query, manifestos)

for section in mapped_sections:
    print(section)

def generate_response(mapped_sections):
    if mapped_sections:
        response = "\n".join(mapped_sections)
    else:
        response = "Sorry, I couldn't find information on that topic in the party manifesto."
    return response

# Example user query
user_query = "What are BJP's plans for women?"

# Map the user query to relevant sections in the party manifestos
mapped_sections = map_query_to_manifesto(user_query, manifestos)

# Generate response based on the mapped sections
chatbot_response = generate_response(mapped_sections)

# Print the chatbot response
print("Chatbot Response:")
print(chatbot_response)
