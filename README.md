# learn_rag_Huggingface

A personal collection of learning notebooks for building **Retrieval-Augmented Generation (RAG)** systems. Each notebook focuses on a different concept, data type, or technique, progressing from core retrieval over unstructured data to multimodal pipelines, agentic workflows, and evaluation.

All notebooks are written for **Google Colab** (they mount Google Drive and read API keys from Colab `userdata`), and use **LangChain**, **FAISS**, and the **OpenAI API** as the core stack. Each one includes an "Open In Colab" badge at the top.

## Notebooks

### 1. `Langchain_and_Unstructured_Data_Video.ipynb` — RAG fundamentals over unstructured data
The foundational notebook. It shows how to take messy, unstructured source files (Excel, Word, etc.) and turn them into a working RAG pipeline:
- Loading and parsing documents with the `unstructured` library
- Chunking text with `RecursiveCharacterTextSplitter`
- Creating embeddings (`text-embedding-3-small`) and storing them in a **FAISS** vector store
- Wrapping the retrieval + generation steps into reusable helper functions

**Start here** if you're new to RAG.

### 2. `RAGAS_Video_1.ipynb` — Evaluating a RAG system
Builds a RAG pipeline over live web documentation (Kubernetes docs loaded via `UnstructuredURLLoader`) and then focuses on **measuring quality** using the **RAGAS** framework. Covers loading/splitting documents, building the vector store, running RAG queries, and scoring the results so you can quantify retrieval and answer quality instead of eyeballing it.

### 3. `Agentic_RAG.ipynb` — Agentic RAG with LangGraph
Moves beyond single-shot retrieval into **agentic RAG**, where an LLM-driven agent manages a multi-turn conversation and decides how to act. Uses **LangGraph** to define an explicit agent state machine (`AgentState`) that tracks conversation turns, questions, and answers, retrieving from a restaurant-menu vector store as needed. A practical intro to stateful, multi-step RAG agents.

### 4. `MultiModal_RAG_Video.ipynb` — Multimodal RAG (video → audio → text)
Demonstrates how to bring **non-text media** into a RAG system. Takes a video file, extracts and compresses the audio with `ffmpeg`/`pydub`, transcribes it, and feeds the resulting text into the retrieval pipeline — showing how to make audio/video content searchable.

### 5. `Capstone_Project_Multimodal_Data_Starter_File.ipynb` — Multimodal capstone
A larger end-to-end project that combines several techniques. It transcribes audio with **OpenAI Whisper** (`large-v3-turbo`), embeds the transcript with `sentence-transformers`, and works with PDFs (`pdf2image` / `poppler`) to build a multimodal RAG system over a real-world finance dataset (Starbucks earnings). Serves as a starter file to practice integrating the concepts from the earlier notebooks.

## Suggested learning order

1. `Langchain_and_Unstructured_Data_Video.ipynb` (core RAG)
2. `RAGAS_Video_1.ipynb` (evaluation)
3. `Agentic_RAG.ipynb` (agents)
4. `MultiModal_RAG_Video.ipynb` (multimodal)
5. `Capstone_Project_Multimodal_Data_Starter_File.ipynb` (capstone)

## Tech stack

- **LangChain** / **LangGraph** — orchestration and agents
- **FAISS** — vector store
- **OpenAI API** — LLMs and embeddings
- **RAGAS** — RAG evaluation
- **Whisper**, **sentence-transformers**, **unstructured**, **pdf2image** — multimodal / data ingestion

## Running the notebooks

These are built for **Google Colab**. To run one:
1. Open it via the "Open In Colab" badge at the top of the notebook.
2. Mount your Google Drive and adjust the `%cd` path to where your data lives.
3. Add the required API key(s) to Colab's **Secrets** (`userdata`) — the notebooks read keys such as `OPENAI_API_KEY` from there.

## Roadmap

🚧 **Coming soon:** notebooks dedicated to learning **Hugging Face** — using Hugging Face models, embeddings, datasets, and the `transformers` ecosystem within RAG pipelines (as an alternative/complement to the OpenAI-based stack used here).
