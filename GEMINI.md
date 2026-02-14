# Directory Overview

This directory contains a comprehensive course on "MongoDB RAG AI with Voyage-AI". The course is structured as a series of Marp-compatible Markdown files, designed for a 3-hour beginner-level workshop. It covers concepts, theory, and hands-on labs.

## Key Files

- `COURSE.md`: The main file containing all 100 slides for the course. It's designed to be run with Marp.

- `labs/`: This directory contains Jupyter Notebooks that guide students through the practical labs outlined in the course.
  - `labs/lab1_mongodb_atlas_setup.ipynb`: Guides through setting up a MongoDB Atlas cluster.
  - `labs/lab2_data_ingestion.ipynb`: Demonstrates data ingestion into MongoDB with Voyage-AI embeddings.
  - `labs/lab3_rag_query_flow.ipynb`: Details the RAG query flow using Voyage-AI and MongoDB Vector Search.
  - `labs/lab4_reranking.ipynb`: Explains how to enhance RAG with Voyage-AI reranking.

## Usage

1. set up `npm install -g @marp-team/marp-cli`
2. run this command `marp -s ./`