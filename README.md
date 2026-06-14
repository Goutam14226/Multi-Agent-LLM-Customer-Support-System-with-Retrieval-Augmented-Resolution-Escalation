# Retrieval-Augmented Customer Support Chatbot with Intelligent Escalation

An AI-powered customer support chatbot that combines **Retrieval-Augmented Generation (RAG)**, **Large Language Models (LLMs)**, **conversation memory**, and **uncertainty-aware escalation** to deliver accurate, context-aware, and reliable customer support responses.

## Overview

Traditional customer support systems often struggle with scalability, consistency, and response accuracy. This project addresses these challenges by integrating a Retrieval-Augmented Generation (RAG) pipeline with a Large Language Model to generate responses grounded in company policy documents.

The system evaluates the reliability of each generated response using multiple confidence signals. Queries with low confidence are automatically escalated to the appropriate support department through a BERT-based routing model.

## Features

* Retrieval-Augmented Generation (RAG) for knowledge-grounded responses
* Pinecone vector database for semantic document retrieval
* LLaMA 3.1 8B Instruct for response generation
* Multi-turn conversation support using memory
* Retrieval confidence evaluation
* Entropy-based uncertainty estimation
* Perplexity-based confidence estimation
* Automatic escalation of uncertain queries
* BERT-based department routing
* Reduced hallucinations through context grounding

## System Architecture

```text
User Query
    │
    ▼
Query Embedding
    │
    ▼
Pinecone Vector Database
    │
    ▼
Top-K Relevant Chunks
    │
    ▼
Prompt Augmentation
(Query + Context + History)
    │
    ▼
LLaMA 3.1 Instruct
    │
    ▼
Generated Response
    │
    ▼
Confidence Evaluation
(Similarity + Entropy + Perplexity)
    │
 ┌──┴──┐
 │     │
 ▼     ▼
Respond  Escalate
            │
            ▼
      BERT Routing
            │
            ▼
 Appropriate Department
```

## Project Structure

```text
├── Data/
│   ├── extracted_documents.json
│   ├── cleaned_documents.json
│   ├── chunked_documents.json
│   └── embeddings.json
│
├── Notebooks/
│   ├── BiText_Data_Preprocessing.ipynb
│   ├── RAG_Database_Preparation.ipynb
│   ├── Testing_Pinecone_connection.ipynb
│   ├── Training_BERT_Department_Routing.ipynb
│   └── Testing_BERT_Routing.ipynb
│
├── Models/
│   ├── LLaMA 3.1 Instruct
│   └── BERT Routing Model
│
└── README.md
```

## Technologies Used

* Python
* LangChain
* Pinecone
* Hugging Face Transformers
* Sentence Transformers
* LLaMA 3.1 8B Instruct
* BERT
* PyTorch
* Pandas
* NumPy

## RAG Pipeline

### 1. Document Processing

* Extract text from customer support policy documents
* Clean and preprocess text
* Split documents into semantic chunks
* Generate embeddings for each chunk

### 2. Vector Storage

* Store embeddings in Pinecone
* Maintain metadata such as source, page number, and chunk ID

### 3. Retrieval

* Convert user query into embeddings
* Perform semantic similarity search
* Retrieve Top-K relevant chunks

### 4. Response Generation

* Combine:

  * User Query
  * Retrieved Context
  * Conversation History

* Generate response using LLaMA 3.1 Instruct

### 5. Reliability Check

The system evaluates:

* Retrieval Similarity Score
* Entropy
* Perplexity

Low-confidence responses are escalated automatically.

### 6. Department Routing

Escalated queries are classified into departments such as:

* Refund
* Return
* Order
* Delivery
* Payment
* Contact Support

using a BERT-based classifier.

## Example

```python
query = "My refund has not been processed yet."

docs = retriever.search(query)

response = llm.generate(
    query=query,
    context=docs
)

if confidence_score < threshold:
    department = router.predict(query)
    print(f"Escalated to {department}")
else:
    print(response)
```

## Key Findings

* RAG significantly improves factual correctness.
* Retrieval grounding reduces hallucinations.
* Fine-tuning on poorly aligned customer-support datasets can degrade performance.
* Instruction-tuned models provide better real-world generalization.
* Combining retrieval confidence and model uncertainty improves system reliability.

## Future Improvements

* Hybrid Retrieval (BM25 + Dense Retrieval)
* Advanced memory management
* Human feedback integration
* Real-time support dashboard
* Multi-language customer support
* Agentic workflow with multiple specialized agents

## Authors

* Goutam Agarwal
* Shoumik Das
* Sandipan Mondal

## Course Information

**IE 624 – Generative and Agentic AI**
**Industrial Engineering and Operations Research**
**IIT Bombay**
