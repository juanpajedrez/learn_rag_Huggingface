# learn_rag_Huggingface

A personal collection of learning notebooks for building **Retrieval-Augmented Generation (RAG)** systems, plus hands-on **Hugging Face** projects covering fine-tuning, structured data extraction, and object detection. Each notebook focuses on a different concept, data type, or technique, progressing from core retrieval over unstructured data to multimodal pipelines, agentic workflows, evaluation, and now Hugging Face model fine-tuning and computer vision.

All RAG notebooks are written for **Google Colab** (they mount Google Drive and read API keys from Colab `userdata`), and use **LangChain**, **FAISS**, and the **OpenAI API** as the core stack. Each one includes an "Open In Colab" badge at the top.

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

### 6. `HuggingFace_Small_LLM.ipynb` — Fine-tuning a small LLM for structured data extraction

The first notebook in the Hugging Face track. Fine-tunes a small, efficient LLM from the Hugging Face ecosystem to pull structured fields out of unstructured text — the same technique behind the **FoodExtract** demo below, which extracts structured nutrition/food data from raw text using a fine-tuned model.

**Live demo:** [FoodExtract — Fine-tuned LLM Structured Data Extractor v1](https://huggingface.co/spaces/juanpajedrez/FoodExtract-v1-VIDEO)

### 7. `learn_hf_object_detection_video.ipynb` — Object detection with RT-DETR-V2

Walks through using **RT-DETR-V2**, a real-time transformer-based object detection model from Hugging Face, to detect and localize objects in video frames — covering model loading, inference, and drawing bounding boxes over detected classes.

### 8. `learn_hf_object_detection_gradio_app.ipynb` — Shipping the detector as a Gradio app

Takes the RT-DETR-V2 object detection workflow from the notebook above and wraps it in an interactive **Gradio** app, ready to deploy as a Hugging Face Space — turning a notebook experiment into a shareable, click-and-try demo.

**Live demo:** [Trashify — RT-DETR-V2 object detection demo 🚮](https://huggingface.co/spaces/juanpajedrez/trashify_demo_video)

## Suggested learning order

**RAG track:**
1. `Langchain_and_Unstructured_Data_Video.ipynb` (core RAG)
2. `RAGAS_Video_1.ipynb` (evaluation)
3. `Agentic_RAG.ipynb` (agents)
4. `MultiModal_RAG_Video.ipynb` (multimodal)
5. `Capstone_Project_Multimodal_Data_Starter_File.ipynb` (capstone)

**Hugging Face track:**
6. `HuggingFace_Small_LLM.ipynb` (fine-tuning for structured extraction)
7. `learn_hf_object_detection_video.ipynb` (object detection fundamentals)
8. `learn_hf_object_detection_gradio_app.ipynb` (deploying as a Gradio Space)

## Tech stack

- **LangChain** / **LangGraph** — orchestration and agents
- **FAISS** — vector store
- **OpenAI API** — LLMs and embeddings
- **RAGAS** — RAG evaluation
- **Whisper**, **sentence-transformers**, **unstructured**, **pdf2image** — multimodal / data ingestion
- **Hugging Face `transformers`** — model fine-tuning and inference
- **RT-DETR-V2** — real-time object detection
- **Gradio** — interactive demo apps / Hugging Face Spaces

## Live demos

- 🍽️ [FoodExtract v1 — Fine-tuned LLM Structured Data Extractor](https://huggingface.co/spaces/juanpajedrez/FoodExtract-v1-VIDEO)
- 🚮 [Trashify — RT-DETR-V2 Object Detection Demo](https://huggingface.co/spaces/juanpajedrez/trashify_demo_video)

## Running the notebooks

These are built for **Google Colab**. To run one:

1. Open it via the "Open In Colab" badge at the top of the notebook.
2. Mount your Google Drive and adjust the `%cd` path to where your data lives.
3. Add the required API key(s) to Colab's **Secrets** (`userdata`) — the notebooks read keys such as `OPENAI_API_KEY` from there.

## Roadmap

✅ ~~Notebooks dedicated to learning Hugging Face~~ — done! See notebooks 6–8 above and the live demos.

🚧 **Coming soon:** deeper dives into Hugging Face `datasets`, additional fine-tuning tasks, and combining Hugging Face models directly into the RAG pipelines from the earlier notebooks.

## License

MIT
