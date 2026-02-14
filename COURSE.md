---
marp: true
theme: default
class: lead
paginate: true
backgroundColor: #89f0a5
---

<style>
{ 
  font-size: 20px;
  font-family: Tahoma;
}
section a {
  color: #ff6600; /* Orange links */
}
</style>

<!-- markdownlint-disable MD025 MD024 MD026 -->

<!-- _class: lead -->
# MongoDB RAG AI with Voyage-AI

## A Beginner's Guide to Retrieval Augmented Generation

---

<!-- _class: heading -->
## Learning Objectives

- Understand core AI, LLM, and RAG concepts.
- Explore MongoDB's role in RAG with Vector Search.
- Discover how Voyage-AI enhances RAG.
- Build a basic RAG application step-by-step.

---

<!-- _class: heading -->

## Piti Champeethong (Fyi)

[Thailand MongoDB User Group](https://www.facebook.com/groups/104161329726941)
[PyLanna - Python User Group](https://www.facebook.com/groups/1181951933835645)

---

## What is Artificial Intelligence (AI)?

- **AI** is the simulation of human intelligence processes by machines.
- These processes include learning, reasoning, and self-correction.
- Goal: Create systems that can think and act like humans.

---

## Brief History of AI

- **1950s:** Concept coined, early programs (e.g., Chess players).
- **1980s:** Expert Systems.
- **2000s:** Machine Learning gains traction.
- **2010s:** Deep Learning revolution, big data, powerful hardware.
- **Today:** AI is everywhere! (Siri, Netflix, self-driving cars)

---

<!-- _class: heading -->

## Types of AI

- **Narrow AI (ANI):** Designed for a specific task.
  - *Examples:* Chess programs, spam filters, recommendation systems.
  - Most AI we interact with today is Narrow AI.

- **General AI (AGI):** Human-level intelligence across all tasks.
  - *Theoretical:* No true AGI exists yet.

---

<!-- _class: heading -->
## Machine Learning (ML) Basics

- A subset of AI.
- Systems learn from data without explicit programming.
- **Supervised Learning:** Learn from labeled data (e.g., cat/dog images).
- **Unsupervised Learning:** Find patterns in unlabeled data (e.g., customer segmentation).

---

<!-- _class: heading -->
## Deep Learning (DL)

- A subset of Machine Learning.
- Uses **Neural Networks** with many layers ("deep").
- Inspired by the human brain's structure.
- Excellent for complex patterns: image recognition, natural language.

---

<!-- _class: heading -->
## What are Large Language Models (LLMs)?

- A type of Deep Learning model.
- Trained on vast amounts of text data (books, articles, internet).
- Can understand, generate, and translate human-like text.
- *Examples:* ChatGPT, Google Gemini, Claude.

---

## How do LLMs Work (Conceptually)?

- Learn patterns, grammar, and facts from text.
- Predict the next word in a sequence.
- Generate coherent and contextually relevant responses.
- It's like a highly sophisticated auto-complete!

---

<!-- _class: heading -->
## The Problem with LLMs

- **Hallucinations:** Can generate factually incorrect but convincing information.
- **Stale Data:** Knowledge is limited to their training cutoff date.
- **Lack of Specificity:** Struggle with niche or proprietary knowledge.
- **Confidence in Errors:** Present false info confidently.

---

<!-- _class: heading -->

## Introduction to Retrieval Augmented Generation (RAG)

- **RAG** combines LLMs with external knowledge sources.
- Solves the problems of hallucinations and stale data.
- Provides LLMs with *real-time, verifiable, and domain-specific information*.

---

## RAG: The Analogy

- Imagine an LLM as a brilliant, creative student.
- But they can't remember everything!
- RAG is like giving them access to a perfectly organized library *during an exam*.
- They retrieve the right book (context) then answer the question.

---

## RAG: Core Idea

1. **Retrieve:** Find relevant information from a trusted data source.
2. **Augment:** Add this information to the LLM's prompt.
3. **Generate:** The LLM uses this augmented prompt to create a more accurate and informed response.

---

## Why RAG is Powerful

- **Fresh Data:** Keeps LLMs up-to-date with new information.
- **Reduced Hallucinations:** Grounds responses in facts.
- **Domain-Specific:** Uses your specific business or personal data.
- **Transparency:** Can cite sources (if designed to).

---

<!-- _class: heading -->
## RAG Workflow (Step 1: Indexing)

- **Prepare your data:**
  - Break down large documents into smaller "chunks."
  - Convert these chunks into numerical representations called "embeddings" (vectors).
  - Store these embeddings in a searchable database (vector database).

---

## RAG Workflow (Step 2: Retrieval)

- **User asks a question.**
- Convert the user's question into an embedding.
- Search the vector database for chunks whose embeddings are "most similar" to the question's embedding.

---

## RAG Workflow (Step 3: Augmentation)

- Take the retrieved relevant chunks of information.
- Combine them with the original user question.
- This creates an "augmented prompt" for the LLM.
  - *Example:* "Based on the following context: [retrieved info], answer: [user question]"

---

## RAG Workflow (Step 4: Generation)

- Send the augmented prompt to the LLM.
- The LLM uses the provided context to formulate a precise and accurate answer.
- The LLM's response is grounded in the retrieved facts.

---

<!-- _class: heading -->
## Visualizing the RAG Workflow

---

![RAG Workflow](mdb-ai-rag.svg)

---

<!-- _class: heading -->
## Key Components of RAG

1. **Data Source:** Where your knowledge lives (documents, databases, web).
2. **Retriever:** Turns queries into embeddings, searches vector store.
3. **Vector Store:** Stores document embeddings for fast similarity search.
4. **LLM:** Generates the final response based on augmented prompt.

---

<!-- _class: heading -->
## Embeddings: The Secret Sauce

- Numerical representations (vectors) of text.
- Capture semantic meaning.
- Words/phrases with similar meanings have similar vector representations.
- Essential for finding "relevant" information.

---

## Vector Databases

- Specialized databases to store and efficiently search vector embeddings.
- Allow for "similarity search" or "nearest neighbor search."
- Essential for fast retrieval in RAG systems.

---

<!-- _class: heading -->
## RAG vs. Fine-tuning

- **RAG:**
  - Updates LLM knowledge dynamically.
  - Cost-effective for frequent knowledge updates.
  - Easier to implement.
  - "Open-book exam" approach.

- **Fine-tuning:**
  - Permanently adjusts LLM's weights with new data.
  - Good for adapting LLM's *style* or *format*.
  - Costly, time-consuming.
  - "Memorizing new facts" approach.

---

<!-- _class: heading -->

## Advantages & Challenges of RAG

---

## Advantages of RAG

- **Accuracy:** Grounds LLM responses in facts.
- **Freshness:** Easily update knowledge by updating the vector store.
- **Cost-Effective:** Don't need to retrain LLM.
- **Transparency:** Potential to show sources.
- **Reduced Hallucinations.**

---

## Disadvantages & Challenges of RAG

- **Retrieval Quality:** If the retriever fails, the LLM fails.
- **Latency:** Extra steps (embedding, search) add time.
- **Context Window Limits:** LLMs have limits on input size.
- **Data Chunking Strategy:** Important for effective retrieval.
- **Complexity:** Managing data, embeddings, vector store, LLM.

---

<!-- _class: lead -->
# Module 2: MongoDB for RAG

## Leveraging MongoDB Atlas Vector Search

---

<!-- _class: heading -->
# Introduction to MongoDB

---

## What is MongoDB?

- A popular NoSQL, document-oriented database.
- Stores data in flexible, JSON-like documents.
- Unlike traditional relational databases (SQL), no fixed schema.
- Built for scalability and high performance.

---

## Why MongoDB for RAG?

- **Flexibility:** Easily store diverse data types (text, images, metadata).
- **Scalability:** Handles large volumes of data and traffic.
- **JSON Documents:** Natural fit for unstructured and semi-structured data.
- **Integrated Vector Search:** A dedicated feature for RAG.

---

<!-- _class: heading -->
# MongoDB Atlas

---

## What is MongoDB Atlas?

- MongoDB's fully managed cloud database service.
- Simplifies deployment, operation, and scaling of MongoDB.
- Available on AWS, Google Cloud, Azure.
- Essential for accessing advanced features like Vector Search.

---

## Key Features for AI/RAG

- **Flexible Schema:** Evolve your data model as RAG needs change.
- **Aggregation Pipeline:** Powerful data transformation capabilities.
- **Atlas Search:** Full-text search engine.
- **Atlas Vector Search:** The core component for RAG retrieval.

---

<!-- _class: heading -->
# Vector Search in MongoDB Atlas

---

## What are Embeddings (Recap)?

- Numerical representations (vectors) of text.
- Capture semantic meaning.
- Used to find similar pieces of information.

---

## Vector Search: The Game Changer

- MongoDB Atlas now includes a native Vector Search capability.
- Allows you to store vector embeddings directly in your database.
- Perform efficient similarity searches on these vectors.
- Eliminates the need for a separate vector database.

---

## How Vector Search Works (Concept)

- When you query with an embedding, Atlas finds documents with "closest" embeddings.
- This "closeness" is measured by distance metrics (e.g., cosine similarity).
- Think of it as finding documents that are semantically similar to your query.

---

## Creating Vector Embeddings

- Before storing, you need to generate embeddings for your data.
- Use an **embedding model** (e.g., from Voyage-AI, OpenAI, Hugging Face).
- This model takes your text and outputs a vector (array of numbers).

---

## Storing Vectors in MongoDB

- Simply add a new field to your existing documents to store the embedding.

```json
{
  "_id": "doc1",
  "text_chunk": "MongoDB is a NoSQL database.",
  "embedding": [0.1, 0.5, -0.2, ..., 0.8]
}
```

---

<!-- _class: heading -->
# MongoDB Atlas Vector Search Index

---

## Creating an Atlas Vector Search Index

- To enable fast vector searches, you need to create a dedicated index.
- This index tells MongoDB which field contains the vectors and how to search them.
- Can be created via Atlas UI or programmatically.

---

## Anatomy of an Atlas Vector Search Index

- **`fields`**: Specifies the vector field and its properties.
  - `path`: The name of your embedding field (e.g., `embedding`).
  - `numDimensions`: The size of your vector (e.g., 1536, 1024).
  - `similarity`: The distance metric (`dotProduct`, `cosine`, `euclidean`).
- Other settings like `type` (HNSW is common for vector search).

---

## Example: Vector Search Index Definition

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "numDimensions": 1536,
        "similarity": "cosine"
      }
    }
  }
}
```

---

<!-- _class: heading -->
# Querying with Vector Search

---

## Querying with Vector Search

- Use the `$vectorSearch` aggregation pipeline stage.
- Takes your query embedding, the path to your vector field, and the number of results.
- Returns documents ordered by similarity.

---

## Example: Simple Vector Search Query (Conceptual)

```python
collection.aggregate([
  {
    '$vectorSearch': {
      'queryVector': [0.1, 0.2, ...], # your query embedding
      'path': 'embedding',
      'numCandidates': 100, # how many to check
      'limit': 10 # how many to return
    }
  }
])
```

---

## Filtering with Vector Search

- Combine `$vectorSearch` with standard MongoDB query operators.
- This allows you to filter documents *before* or *after* the vector search.
- Example: Find similar documents *only* from a specific category.

---

## Example: Vector Search with Filtering

```python
collection.aggregate([
  {
    '$vectorSearch': {
      'queryVector': [...],
      'path': 'embedding',
      'numCandidates': 100,
      'limit': 10,
      'filter': { 'category': 'science' } # Add a filter!
    }
  }
])
```

---

## Hybrid Search

- Combines keyword-based search (Atlas Search) and vector search.
- Benefits:
  - Keyword search is good for exact matches.
  - Vector search is good for semantic understanding.
- Often yields superior retrieval results for RAG.

---

<!-- _class: heading -->
# Aggregation Pipeline for RAG

- The MongoDB Aggregation Pipeline is key for building a full RAG flow.
- Chaining stages:
  1. `$vectorSearch` to find similar documents.
  2. `$project` to reshape results.
  3. `$limit` to control output.
  4. And more!

---

## Example: RAG Aggregation Pipeline (Conceptual)

```python
collection.aggregate([
  { '$vectorSearch': { ... } },      # 1. Retrieve by similarity
  { '$project': {                    # 2. Select relevant fields
      'text_chunk': 1,
      'score': { '$meta': 'vectorSearchScore' }
    }
  },
  { '$limit': 5 }                     # 3. Take top 5
])
```

---

<!-- _class: heading -->
# Data Ingestion & Best Practices

---

## Data Ingestion for RAG in MongoDB

- **Process:**
  1. Load raw data.
  2. Chunk text into manageable sizes.
  3. Generate embeddings for each chunk using an embedding model.
  4. Insert documents (with `text_chunk` and `embedding`) into MongoDB.

---

## Updating Data in RAG

- Simply update or insert new documents into your collection.
- Atlas Vector Search automatically keeps the index synchronized.
- Ensures your RAG application always has the latest context.

---

## Scalability with MongoDB Atlas

- Easily scale your cluster horizontally (add more nodes).
- Handles growing data volumes and query loads.
- Essential for production RAG applications.

---

## Security in MongoDB Atlas

- Robust security features built-in:
  - Network isolation.
  - Authentication (SCRAM, LDAP, x.509).
  - Encryption at rest and in transit.
  - Role-based access control.

---

## Best Practices for MongoDB RAG

- **Chunking Strategy:** Experiment with chunk size and overlap.
- **Embedding Model Choice:** Select a model suited to your data and task.
- **Index Tuning:** Optimize `numCandidates` for performance vs. recall.
- **Hybrid Search:** Combine vector and keyword for best results.
- **Data Modeling:** Store metadata with chunks for effective filtering.

---

## MongoDB Atlas for Production RAG

- **Reliability:** Built-in backups, high availability.
- **Observability:** Monitoring, performance advisor.
- **Developer Data Platform:** Integrates with other services (Functions, Triggers).

---

## Alternatives to Vector Search (Brief Mention)

- While Vector Search is ideal, you could technically:
  - Store embeddings in a separate dedicated vector database.
  - Implement basic similarity search manually (less efficient).
  - Use only full-text search (less semantic understanding).

---

<!-- _class: lead -->
# Module 3: Introduction to Voyage-AI

## Enhancing RAG with Powerful Embeddings & Reranking

---

<!-- _class: heading -->
# What is Voyage-AI?

---

## What is Voyage-AI?

- A leading platform specializing in **high-performance embedding models** and **rerankers**.
- Designed to empower developers to build smarter AI applications, especially for RAG.
- Provides robust APIs for converting text into meaningful numerical representations.

---

## Why use Voyage-AI?

- **Accuracy:** State-of-the-art models for superior semantic understanding.
- **Speed:** Optimized for fast embedding generation.
- **Cost-effectiveness:** Efficient models can reduce operational costs.
- **Ease of Use:** Simple APIs and SDKs for quick integration.
- **Reranking:** Advanced capabilities to refine search results.

---

<!-- _class: heading -->
# Voyage-AI Models

---

## Voyage-AI Embedding Models

- Offers a range of models tailored for different use cases.
- **General-purpose models:** Good for broad applications.
- **Specialized models:** Optimized for specific domains or tasks.
- Key function: Transform text into high-quality vector embeddings.

---

## Key Features of Voyage-AI

1. **Embeddings API:** For generating numerical vectors from text.
2. **Reranking API:** To improve the relevance of retrieved documents.
3. **Python SDK:** Easy integration into Python applications.

---

<!-- _class: heading -->
# Getting Started with Voyage-AI

---

## Getting Started with Voyage-AI

1. **Obtain an API Key:** Sign up on the Voyage-AI platform to get your unique key.
2. **Install the Python SDK:**

    ```bash
    pip install voyageai
    ```

3. **Initialize Client:**

    ```python
    import voyageai
    vo = voyageai.Client(api_key="YOUR_VOYAGE_API_KEY")
    ```

---

## Generating Embeddings with Voyage-AI

- Use the `vo.embed()` function.
- Input: List of text strings.
- Output: List of corresponding vector embeddings.

---

## `voyage.embed()` function

```python
response = vo.embed(
    texts=["Hello, world!", "Voyage AI is great."],
    model="voyage-3-large", # or other models
    input_type="document" # or "query" for search
)
embeddings = response.embeddings
```

---

## Example: Text Embedding

```python
import voyageai
import os

# Ensure your API key is set as an environment variable
# vo = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
# For demo, replace with actual key or ensure env var is set
vo = voyageai.Client(api_key="YOUR_VOYAGE_API_KEY")

texts_to_embed = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is rapidly advancing.",
    "Retrieval Augmented Generation improves LLM accuracy."
]

try:
    result = vo.embed(texts=texts_to_embed, model="voyage-3-large")
    for i, embedding in enumerate(result.embeddings):
        print(f"Embedding for text '{texts_to_embed[i]}' has dimension {len(embedding)}")
except Exception as e:
    print(f"Error generating embeddings: {e}")
```

---

## Batching Embeddings

- For large datasets, batching is crucial for efficiency.
- `vo.embed()` can take a list of many texts.
- Reduces API calls and improves throughput.

---

<!-- _class: heading -->
# Voyage-AI for Reranking

---

## Voyage-AI for Reranking

- After initial retrieval, some documents might be less relevant.
- Reranking re-scores a set of retrieved documents based on their relevance to the query.
- Improves the quality of the context provided to the LLM.

---

## How Reranking Works (Concept)

1. **Initial Retrieval:** Get top-N documents based on vector similarity.
2. **Reranking:** Pass these N documents and the query to a reranker model.
3. **Rescore:** The reranker provides new relevance scores.
4. **Reorder:** Sort documents by new scores to present the most relevant ones.

---

## `voyage.rerank()` function

```python
query = "What is the capital of France?"
documents = [
    "The Eiffel Tower is in Paris.",
    "Berlin is the capital of Germany.",
    "Paris is the capital and most populous city of France."
]

rerank_result = vo.rerank(query=query, documents=documents, model="rerank-lite-1")
# rerank_result.results contains documents with new relevance scores
```

---

## Benefits of Reranking in RAG

- **Higher Precision:** More accurate context for the LLM.
- **Better User Experience:** LLM provides more relevant answers.
- **Optimized LLM Usage:** Reduces wasted LLM tokens on irrelevant context.

---

<!-- _class: lead -->
# Module 4: Building a RAG Application

## Workshop: MongoDB Atlas & Voyage-AI in Action

---

<!-- _class: heading -->
# Workshop Introduction

---

## What We'll Build

- A simple RAG application.
- Uses MongoDB Atlas for vector storage and search.
- Leverages Voyage-AI for embedding generation and (optional) reranking.
- Integrates with an LLM (conceptually, e.g., OpenAI, Gemini).

---

## Prerequisites

- **Python 3.8+** installed.
- **MongoDB Atlas Account:** Free tier is sufficient.
- **Voyage-AI API Key:** Sign up at voyageai.com.
- **LLM API Key (Optional):** E.g., OpenAI API key for actual generation.

---

## Setting up Your Environment

1. **Create a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2. **Install necessary libraries:**

    ```bash
    pip install pymongo voyageai python-dotenv openai # Or other LLM library
    ```

3. **Create a `.env` file** (for API keys):

```python
    VOYAGEAI_API_KEY="your_voyage_ai_key"
    MONGODB_URI="your_mongodb_atlas_connection_string"
    AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
    AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
    AZURE_OPENAI_DEPLOYMENT_NAME="your_azure_openai_deployment_name"
    AZURE_OPENAI_API_VERSION="your_azure_openai_api_version"
```

---

<!-- _class: heading -->
# Lab 1: MongoDB Atlas Setup

---

## Lab 1: MongoDB Atlas Setup

1. **Create a Free Tier Cluster:**
    - Go to cloud.mongodb.com.
    - Sign up/Log in.
    - Click "Build a Database" and choose a "Shared" (free) cluster.
    - Select your preferred cloud provider and region.
    - Name your cluster.

---

## Lab 1: Create Database User & Network Access

1. **Create a Database User:**
    - Navigate to "Database Access" under Security.
    - Click "Add New Database User".
    - Choose "Password" authentication, set a strong password.
    - Grant "Read and write to any database".
2. **Configure Network Access:**
    - Navigate to "Network Access" under Security.
    - Click "Add IP Address".
    - For simplicity, choose "Allow Access from Anywhere" (`0.0.0.0/0`).
    - *(Note: For production, restrict to specific IPs.)*

---

## Lab 1: Get Connection String

1. **Connect to Your Cluster:**
    - Go to "Databases" -> "Connect".
    - Choose "Drivers".
    - Select Python, version 3.6 or later.
    - Copy the connection string.
    - **Replace `<password>` with your database user's password.**
    - Add this to your `.env` file as `MONGODB_URI`.

---

## Lab 1: Create a Database and Collection

- We'll create a database named `rag_db` and a collection `documents` programmatically.
- This collection will store our text chunks and their embeddings.

---

<!-- _class: heading -->
# Lab 2: Data Ingestion

---

## Lab 2: Data Preparation

- For this lab, let's use a small set of example text.
- Imagine these are snippets from your company's documentation.

```python
sample_texts = [
    "The new product features include enhanced security protocols and faster processing.",
    "Our customer support is available 24/7 via live chat and email.",
    "This document outlines the privacy policy regarding user data collection and usage.",
    "Upcoming software updates will introduce a dark mode and custom themes.",
    "Please refer to the user manual for detailed installation instructions."
]
```

---

## Lab 2: Step 1: Load Data & Chunk (Conceptual)

- For larger datasets, you'd chunk your data into smaller, meaningful segments.
- For our `sample_texts`, each item is already a chunk.

```python
# In a real application, you'd load from files, APIs, etc.
# For now, our `sample_texts` list acts as our chunks.
```

---

## Lab 2: Step 2: Generate Embeddings

- Use Voyage-AI to generate embeddings for each text chunk.

```python
from dotenv import load_dotenv
import os
import voyageai
from pymongo import MongoClient

load_dotenv() # Load environment variables from .env

vo = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))

sample_texts = [
    "The new product features include enhanced security protocols and faster processing.",
    "Our customer support is available 24/7 via live chat and email.",
    "This document outlines the privacy policy regarding user data collection and usage.",
    "Upcoming software updates will introduce a dark mode and custom themes.",
    "Please refer to the user manual for detailed installation instructions."
]

print("Generating embeddings with Voyage-AI...")
response = vo.embed(texts=sample_texts, model="voyage-3-large", input_type="document")
embeddings = response.embeddings
print(f"Generated {len(embeddings)} embeddings.")
```

---

## Lab 2: Step 3: Store in MongoDB

- Connect to MongoDB Atlas.
- Insert each text chunk along with its generated embedding into a collection.

---

## Lab 2: Step 3: Store in MongoDB (Continue)

```python
MONGODB_URI = os.environ.get("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client['rag_db']
collection = db['documents']

documents_to_insert = []
for i, text in enumerate(sample_texts):
    documents_to_insert.append({
        "text_chunk": text,
        "embedding": embeddings[i],
        "source": f"doc_{i+1}" # Example metadata
    })

print("Inserting documents into MongoDB...")
if documents_to_insert:
    collection.insert_many(documents_to_insert)
    print(f"Inserted {len(documents_to_insert)} documents.")
else:
    print("No documents to insert.")

client.close()
```

---

## Creating an Atlas Vector Search Index (Lab 2 Continued)

- We need an index on the `embedding` field to perform vector searches.
- You can create this via the Atlas UI:
  - Go to your cluster in Atlas.
  - Click "Search" tab.
  - Click "Create Search Index".
  - Select "JSON Editor" and paste the definition.

---

## Lab 2: Index Definition Example (JSON Editor)

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "numDimensions": 1536,
        "similarity": "cosine"
      }
    }
  }
}
```

- Replace `numDimensions` with the actual dimension of your Voyage-AI embeddings (e.g., 1536 for voyage-3-large).

---

<!-- _class: heading -->
# Lab 3: Building the RAG Query Flow

---

## Lab 3: The RAG Query Flow

1. **User Query:** Get the question from the user.
2. **Embed Query:** Use Voyage-AI to create an embedding of the query.
3. **Vector Search:** Query MongoDB Atlas using `$vectorSearch`.
4. **Retrieve Context:** Extract relevant text chunks from results.
5. **Augment Prompt:** Combine query + context for the LLM.
6. **Generate Response:** Send augmented prompt to LLM.

---

## Lab 3: Step 1: User Query & Embed

```python
# (Assuming MongoClient and vo are initialized as before)
query_text = "Tell me about the upcoming features of the software."
print(f"
User Query: {query_text}")

print("Generating query embedding with Voyage-AI...")
query_embedding_response = vo.embed(
    texts=[query_text],
    model="voyage-3-large",
    input_type="query" # Important for query embeddings
)
query_embedding = query_embedding_response.embeddings[0]
print("Query embedding generated.")
```

---

## Lab 3: Step 2: MongoDB Vector Search

```python
client = MongoClient(os.environ.get("MONGODB_URI"))
db = client['rag_db']
collection = db['documents']

pipeline = [
  {
    '$vectorSearch': {
      'queryVector': query_embedding,
      'path': 'embedding',
      'numCandidates': 50, # Number of documents to scan
      'limit': 3,          # Number of documents to return
      'index': 'default'   # Name of your vector search index
    }
  },
  {
    '$project': {
      'text_chunk': 1,
      'source': 1,
      'score': { '$meta': 'vectorSearchScore' },
      '_id': 0
    }
  }
]

print("Performing vector search in MongoDB Atlas...")
results = list(collection.aggregate(pipeline))
print(f"Retrieved {len(results)} relevant documents.")
client.close()
```

---

## Lab 3: Step 3: Retrieving Top Results & Building Context

```python
context = "".join([doc['text_chunk'] for doc in results])
print("--- Retrieved Context ---")
print(context)
print("------------------------")
```

---

## Lab 3: Step 4: Integrating with an LLM (Conceptual)

- We'll use a placeholder for an LLM call.
- For a real application, you'd use `openai.ChatCompletion.create()`, Google Gemini API, etc.

```python
# Example of a prompt structure for an LLM
llm_prompt = f"""
You are a helpful assistant. Answer the user's question based on the provided context only.
If you cannot find the answer in the context, politely state that the information is not available.

Context:
{context}

Question: {query_text}

Answer:
"""
print("--- LLM Prompt ---")
print(llm_prompt)
print("------------------")

```

---

## Lab 3: Putting it all Together (Full Script Structure)

```python
# full_rag_app.py
from dotenv import load_dotenv
import os
import voyageai
from pymongo import MongoClient
# from openai import OpenAI # Uncomment if using OpenAI

load_dotenv()

# Initialize Voyage-AI Client
vo = voyageai.Client(api_key=os.environ.get("VOYAGEAI_API_KEY"))

# Initialize MongoDB Client
MONGODB_URI = os.environ.get("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client['rag_db']
collection = db['documents']

def run_rag_query(query_text: str):
    # 1. Embed the user query
    query_embedding_response = vo.embed(
        texts=[query_text],
        model="voyage-3-large",
        input_type="query"
    )
    query_embedding = query_embedding_response.embeddings[0]

    # 2. Perform vector search in MongoDB Atlas
    pipeline = [
        {
            '$vectorSearch': {
                'queryVector': query_embedding,
                'path': 'embedding',
                'numCandidates': 50,
                'limit': 3,
                'index': 'default'
            }
        },
        { '$project': { 'text_chunk': 1, 'source': 1, 'score': { '$meta': 'vectorSearchScore' }, '_id': 0 } }
    ]
    results = list(collection.aggregate(pipeline))

    # 3. Build context for the LLM
    context = "".join([doc['text_chunk'] for doc in results])
    if not context:
        return "Sorry, I couldn't find relevant information in my knowledge base."

    # 4. Construct LLM prompt
    llm_prompt = f"""
    You are a helpful assistant. Answer the user's question based on the provided context only.
    If you cannot find the answer in the context, politely state that the information is not available.

    Context:
    {context}

    Question: {query_text}

    Answer:
    """
    
    # 5. Get response from LLM
    # For demonstration, we'll return the prompt.
    # In a real app, you'd call client_openai.chat.completions.create(...)
    # For now, let's simulate LLM response:
    simulated_llm_response = f"LLM would answer this based on the context: '{context[:50]}...' -- {query_text}"
    return simulated_llm_response

if __name__ == "__main__":
    test_query = "What are the latest security features?"
    response = run_rag_query(test_query)
    print(f"Final RAG Response: {response}")
    client.close()
```

---

<!-- _class: heading -->
# Enhancing RAG with Reranking

---

## Enhancing RAG: Reranking with Voyage-AI (Lab 4 Optional)

- To improve precision, especially with more documents.
- Takes the initial search results and re-orders them for higher relevance.

---

## Lab 4: Implementing Reranking

```python
# (Continue from previous script or define necessary objects)

def run_rag_query_with_rerank(query_text: str):
    # 1. Embed the user query
    query_embedding_response = vo.embed(
        texts=[query_text], model="voyage-3-large", input_type="query"
    )
    query_embedding = query_embedding_response.embeddings[0]

    # 2. Perform initial vector search (retrieve more candidates)
    client = MongoClient(os.environ.get("MONGODB_URI"))
    db = client['rag_db']
    collection = db['documents']
    pipeline = [
        {
            '$vectorSearch': {
                'queryVector': query_embedding,
                'path': 'embedding',
                'numCandidates': 100, # Retrieve more candidates for reranking
                'limit': 10,          # Initial set of documents
                'index': 'default'
            }
        },
        { '$project': { 'text_chunk': 1, 'source': 1, 'score': { '$meta': 'vectorSearchScore' }, '_id': 0 } }
    ]
    initial_results = list(collection.aggregate(pipeline))
    client.close()

    if not initial_results:
        return "Sorry, no initial information found."

    # Prepare documents for reranking
    documents_for_rerank = [doc['text_chunk'] for doc in initial_results]
    
    # 3. Rerank the initial results
    rerank_response = vo.rerank(
        query=query_text,
        documents=documents_for_rerank,
        model="rerank-lite-1"
    )
    
    # Sort initial results based on reranked scores
    reranked_documents = [
        documents_for_rerank[r.index] for r in sorted(rerank_response.results, key=lambda x: x.relevance_score, reverse=True)
    ]
    
    # Take top N after reranking
    final_context_chunks = reranked_documents[:3] # Take top 3 reranked docs

    context = "
".join(final_context_chunks)
    
    # 4. Construct LLM prompt & get response (as before)
    llm_prompt = f"""
    You are a helpful assistant. Answer the user's question based on the provided context only.
    ... (rest of prompt) ...
    Context:
    {context}
    Question: {query_text}
    """
    simulated_llm_response = f"LLM (with reranking) would answer: '{context[:50]}...' -- {query_text}"
    return simulated_llm_response

if __name__ == "__main__":
    # ... (previous code) ...
    print("
--- Running RAG with Reranking ---")
    reranked_response = run_rag_query_with_rerank("Tell me about the software's new features.")
    print(f"
Final RAG Response (with Reranking): {reranked_response}")
```

---

## Lab 4: Comparing Results (Before/After Reranking)

- Reranking often yields more coherent and directly relevant context.
- You might observe subtle but impactful differences in LLM responses.
- It's a key optimization for production RAG systems.

---

<!-- _class: heading -->
# Best Practices & Advanced Topics

---

## Best Practices for RAG Applications

- **Smart Chunking:** Don't just split by character; consider semantic units.
- **Prompt Engineering:** Craft clear instructions for the LLM.
- **Metadata Filtering:** Use MongoDB queries to pre-filter documents before vector search.
- **Monitor & Iterate:** Continuously evaluate retrieval quality and LLM responses.

---

## Error Handling and Edge Cases

- **No relevant documents:** Inform the user.
- **LLM API limits/failures:** Implement retries and fallbacks.
- **Context window overflow:** Ensure retrieved context fits LLM limits.

---

## Monitoring and Logging

- Track retrieval latency, LLM response times.
- Log user queries, retrieved documents, and LLM responses for analysis.
- Identify common failure modes or areas for improvement.

---

## Deployment Considerations

- **MongoDB Atlas:** Managed service for scalability and reliability.
- **Application Backend:** FastAPI, Node.js Express, Flask for API endpoints.
- **Serverless:** Atlas App Services, AWS Lambda, Google Cloud Functions.

---

## Advanced Topics (Briefly)

- **Multi-hop RAG:** For complex questions requiring multiple retrievals.
- **Knowledge Graphs:** Structuring data for richer context.
- **Query Rewriting:** Improving user queries before embedding.
- **Self-RAG:** LLM critiques and improves its own retrieval.

---

<!-- _class: heading -->
# Review of Workshop

---

## Review of Workshop

- We set up MongoDB Atlas.
- Ingested data with Voyage-AI embeddings.
- Performed RAG queries.
- Explored (optional) reranking.
- Learned about best practices.

---

## Troubleshooting Common Issues

- **API Key Errors:** Double-check `.env` file and client initialization.
- **MongoDB Connection:** Ensure IP access is configured and password is correct.
- **Index Not Found:** Verify vector search index name in Atlas.
- **No Relevant Results:** Check embedding model choice, chunking, or data quality.

---

## Further Learning Resources

- **MongoDB Developer Center:** [https://www.mongodb.com/resources/](https://www.mongodb.com/resources)
- **Voyage-AI Documentation:** [https://docs.voyageai.com](https://docs.voyageai.com)
- **LangChain / LlamaIndex:** Frameworks for building RAG applications.
- **Online Courses/Tutorials:** Keep learning and building!

---

## Key Takeaways

- RAG is a powerful paradigm for building intelligent, data-aware AI applications.
- MongoDB Atlas and Voyage-AI provide a robust stack for this.
- Start building, experiment, and contribute to the AI community!
