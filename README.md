# Multimodal Cache Augmented Generation System

This project implements a **Cache Augmented Generation System** using **Streamlit**, leveraging **NVIDIA's NIM APIs**, **Qdrant vector database**, and **PyMuPDF** to answer user queries based on PDF documents and image content. The system accelerates responses by caching both embeddings and answers.

## 🚀 Features

- 🔍 **Text-Based Document QA**:
  - Upload a PDF and ask questions about its content.
  - Uses `nv-embedqa-mistral-7b-v2` for embedding and `llama-3.3-nemotron-super-49b-v1` for context-aware response generation.
  - Embeddings stored and retrieved using **Qdrant** (in-memory).

- 🖼️ **Image-Based Analysis**:
  - Upload images and query them using natural language.
  - Uses **NVIDIA VILA model** via API to interpret visual content and answer questions.
  
- 🧠 **Smart Caching**:
  - Uses `pickle` to cache embedding results and final answers.
  - Reduces latency and cost on repeated queries.

- 💾 **On-Demand Cache Clearing**:
  - Button to manually clear all cached responses and embeddings.

## 🧰 Tech Stack

| Component         | Description                                                   |
|------------------|---------------------------------------------------------------|
| Streamlit        | Frontend UI                                                   |
| PyMuPDF (`fitz`) | PDF text extraction                                           |
| OpenAI SDK       | For using NVIDIA's hosted NIM APIs                            |
| Qdrant           | In-memory vector store for fast semantic search               |
| PIL & io         | Image display and byte conversion                             |
| `pickle` & `os`  | Simple file-based caching mechanism                           |

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Lyf-of-sakthi/Multimodal-Cache-Augmented-Generation-System.git
cd Multimodal-Cache-Augmented-Generation-System
