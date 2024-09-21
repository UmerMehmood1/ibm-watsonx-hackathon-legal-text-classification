# Legal Text Classifier using IBM Watsonx Foundation Model

This repository contains code developed for the IBM Watsonx Hackathon. The project aims to classify legal texts into predefined categories, using IBM Watsonx API and Foundation Models.

## Overview

The legal text classifier leverages IBM Watsonx's `granite-13b-chat-v2` model to classify legal documents into categories such as contracts, case law, legislation, and more. This project uses proximity search with a vector index to enhance context-based inferencing.

## Setup

### Prerequisites
Ensure that the following dependencies are installed:
```bash
pip install --upgrade 'chromadb==0.3.26' 'pydantic==1.10.0' sentence-transformers
```
## Watsonx API Connection

To connect to the IBM Watsonx API, you need to provide your IBM Cloud personal API key. This API key will be securely used to authenticate API requests and communicate with the Watsonx services.

### Step 1: IBM Cloud API Key

You can set up the API key by securely inputting it in your Python script as follows:

```python
import os
import getpass

def get_credentials():
    return {
        "url" : "https://us-south.ml.cloud.ibm.com",  # IBM Cloud region endpoint
        "apikey" : getpass.getpass("Please enter your API key (hit enter): ")  # Prompt user for secure input
    }
```
This ensures that the API key is not hardcoded in the script, providing enhanced security when managing sensitive credentials.

### Step 2: Set Model ID and Project Details

You will need to set the `model_id` for the Watsonx model you want to use and retrieve project/space details if necessary.

#### Set Model ID

The `model_id` is critical for specifying which Watsonx model will be used for inferencing:

```python
model_id = "ibm/granite-13b-chat-v2"  # Granite-13B model for large-scale language processing
```
This model identifier links your request to the Watsonx model for inference.

###Retrieve Project/Space Details
If your application is running within a specific IBM Cloud project or space, you can access its identifiers through environment variables:
```python
import os

project_id = os.getenv("PROJECT_ID")  # Retrieve project ID from environment variables
space_id = os.getenv("SPACE_ID")  # Retrieve space ID from environment variables
```

This ensures your API requests are scoped to the correct project or space.

###Step 3: Set Model Parameters
You can control the behavior of the Watsonx model by defining parameters such as decoding method, maximum token generation, and repetition penalties.

Here’s an example of setting the parameters for the model:
```python
parameters = {
    "decoding_method": "greedy",  # Decoding strategy used by the model
    "max_new_tokens": 8191,  # Maximum number of tokens to generate
    "repetition_penalty": 1.05  # Penalty to discourage repetition in the output
}
```
These parameters adjust the model's behavior when processing input text and generating predictions.

## Vector Index for Proximity Search

Vector indexing enables efficient proximity searches, ensuring that the model can quickly find contextually relevant sections of legal documents based on embeddings.

### Step 1: Initialize Vector Index

To perform proximity search, you need to connect to your vector index stored in IBM Watsonx:

```python
from ibm_watsonx_ai.client import APIClient

# Initialize API client using Watsonx credentials
client = APIClient(credentials=wml_credentials, project_id=project_id, space_id=space_id)

# Set the vector index ID
vector_index_id = "1af28755-656b-40db-94f1-b9c74a8509fe"  # Example vector index ID
vector_index_details = client.data_assets.get_details(vector_index_id)  # Retrieve vector index details
```
The vector_index_id is the unique identifier for the specific index you want to work with.

### Step 2: Hydrate ChromaDB with Vector Data

To populate your ChromaDB with vector data from the Watsonx vector index, follow the steps below to fetch and store the embeddings and associated metadata.

#### Fetching Vector Data

This function retrieves the vector data from the Watsonx API, decompresses it, and parses the JSON content to prepare it for storage in ChromaDB:

```python
import gzip
import json
import random
import string
import chromadb

def hydrate_chromadb():
    # Fetch vector index content from Watsonx
    data = client.data_assets.get_content(vector_index_id)
    content = gzip.decompress(data)
    stringified_vectors = content.decode("utf-8")
    vectors = json.loads(stringified_vectors)

    # Initialize ChromaDB client and create a collection
    chroma_client = chromadb.Client()
    collection_name = "legal_vectors"
    
    # Clear existing collection if it exists
    try:
        collection = chroma_client.delete_collection(name=collection_name)
    except:
        print("Collection didn't exist - nothing to do.")
        
    # Create a new collection for storing the vectors
    collection = chroma_client.create_collection(name=collection_name)

    vector_embeddings = []
    vector_documents = []
    vector_metadatas = []
    vector_ids = []

    for vector in vectors:
        vector_embeddings.append(vector["embedding"])  # Store embeddings
        vector_documents.append(vector["content"])      # Store document content

        # Clean and store metadata
        metadata = vector["metadata"]
        lines = metadata["loc"]["lines"]
        clean_metadata = {
            "asset_id": metadata["asset_id"],
            "asset_name": metadata["asset_name"],
            "url": metadata["url"],
            "from": lines["from"],
            "to": lines["to"]
        }
        vector_metadatas.append(clean_metadata)

        # Generate a unique ID for each vector entry
        asset_id = metadata["asset_id"]
        random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        vector_id = f"{asset_id}:{lines['from']}-{lines['to']}-{random_string}"
        vector_ids.append(vector_id)

    # Add all vectors to the ChromaDB collection
    collection.add(
        embeddings=vector_embeddings,
        documents=vector_documents,
        metadatas=vector_metadatas,
        ids=vector_ids
    )
    
    return collection

# Call the function to hydrate ChromaDB
chroma_collection = hydrate_chromadb()
```
This function connects to your Watsonx API, retrieves the vector embeddings, and populates your ChromaDB collection with the necessary data.

### Step 3: Perform Proximity Search

Once your ChromaDB is populated, you can implement a proximity search function to find relevant legal text based on a user query:

```python
def proximity_search(question):
    # Convert the question into an embedding
    query_vectors = emb.embed_query(question)

    # Query the ChromaDB collection for relevant documents
    query_result = chroma_collection.query(
        query_embeddings=query_vectors,
        n_results=vector_index_properties["settings"]["top_k"],  # Adjust the number of results as needed
        include=["documents", "metadatas", "distances"]  # Include document metadata in results
    )

    # Format the results for easy reading
    documents = list(reversed(query_result["documents"][0]))  # Reverse for descending order of relevance
    return "\n".join(documents)
```
This function performs a search in the ChromaDB collection, returning the most relevant documents based on the input query.

### Step 4: Prepare Input for Model Inferencing

To classify legal texts using the foundation model, you need to construct a structured input prompt that provides the necessary context. This prompt includes a system instruction and the relevant documents identified through the proximity search:

```python
prompt_input = """<|system|>
System Prompt: AI Legal Text Classification
Purpose:
You are an AI model designed to classify legal texts into predefined categories to assist legal professionals in document management and retrieval.

Instructions:
Input Types:
Legal documents such as contracts, case law, statutes, legal briefs, etc.
Text snippets or excerpts from larger documents.
Categories for Classification:
- Contracts
- Litigation
- Legislation
- Regulatory Compliance
- Legal Opinions
- Case Law
- Intellectual Property
- Family Law
- Criminal Law
- Corporate Law
Output Requirements:
Format:
Respond using bolded headings with properly indented subsections for the output.

1. Summarization
Provide a brief summary of the input text based on its legal context.
This should explain the core content in concise terms.
2. Classification
Category: The most appropriate legal category for the text.
Confidence Score: Indicate the certainty of the classification (0-100%).
3. Reasoning
Explanation: Why the text is classified into the selected category.
Handling of Ambiguity: If the text contains ambiguous or mixed content, describe how the final classification decision was reached.
Additional Context:
If the input text contains ambiguous or mixed content, discuss any contributing factors in the final classification decision.
Ensure the reasoning clearly explains the connection between the text and the chosen category.
"""
```
This prompt provides the model with clear instructions on how to process the input text and what output format to use.

### Step 5: Execute Model Inferencing

With the prompt constructed, you can now run the model to classify the legal text based on the user input. This is achieved through the following code:

```python
# Get the user's question
question = input("Question: ")

# Fetch relevant documents using proximity search
grounding = proximity_search(question)

# Format the question for the model
formattedQuestion = f"""<|user|>
[Document]
{grounding}
[End]
{question}
<|user|>
"""

# Construct the complete prompt for the model
prompt = f"""{prompt_input}{formattedQuestion}"""

# Generate a response using the model
generated_response = model.generate_text(prompt=prompt.replace("__grounding__", grounding), guardrails=False)
print(f"AI: {generated_response}")
```

### Example Usage

Here’s how you might use the model in practice:

1. **Input a Legal Document:** Provide a legal text or question related to legal texts when prompted.

   Example:
      User: Question: What are the terms of the contract?
      AI: Summarization  
   This text outlines the terms and conditions of a legal contract, specifying the rights and obligations of the parties involved.

   Classification

   - **Category:** Contracts  
   - **Confidence Score:** 90%

   Reasoning

   - **Explanation:** The text discusses a legally binding agreement between parties, fitting the Contracts category.  
   - **Handling of Ambiguity:** The content is clear and directly related to contract terms, with no ambiguous elements present.
   
### Example Usage

1. **Input Document:**  
   ```  
   "This agreement sets forth the terms of service for using the software product. Users must comply with the rules outlined herein."  
   ```

2. **Question: What are the terms of the contract?**  
   ```  
   The model will classify the document and provide a structured response based on the defined categories.  
   ```

3. **Expected Output:**  
   ```  
   1. **Summarization**  
      This text outlines the terms and conditions of a legal contract, specifying the rights and obligations of the parties involved.

   2. **Classification**  
      - **Category:** Contracts  
      - **Confidence Score:** 90%

   3. **Reasoning**  
      - **Explanation:** The text discusses a legally binding agreement between parties, fitting the Contracts category.  
      - **Handling of Ambiguity:** The content is clear and directly related to contract terms, with no ambiguous elements present.
   ```

### Final Notes

This model provides a framework for classifying legal texts, with potential applications in document management and retrieval for legal professionals. Future enhancements may include expanding category definitions and refining model accuracy based on feedback and additional training data.
