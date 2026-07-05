# AI-Powered ATS & Interview Preparation System (RAG-Optimized)

An applied Large Language Model (LLM) application that acts as an automated recruitment screening and interview preparation assistant. The system dynamically processes unstructured PDF resumes, converts them into semantic vector blocks, and matches them against target job descriptions using a Retrieval-Augmented Generation (RAG) pipeline.

## 🚀 Features
* **Dynamic Text Parsing:** Extracts raw string data from uploaded binary PDF files locally in runtime memory.
* **RAG-Powered Semantic Matching:** Chunks data and implements **ChromaDB vector embeddings** to locate exact context matches instead of relying on basic keyword sorting.
* **Semantic Skill-Gap Analysis:** Discovers missing technological requirements and context match percentages accurately using localized matching constraints.
* **Tailored Interview Track:** Automatically generates personalized technical interview question paths based directly on the candidate's resume gaps.
* **Secure Runtime Authentication:** Eliminates hardcoded API credentials by leveraging client-side token entry via the Streamlit interface.

## 🛠️ Tech Stack
* **Language:** Python
* **Vector DB Architecture:** ChromaDB (Retrieval-Augmented Generation)
* **AI Backend:** Google GenAI SDK (`gemini-2.5-flash`)
* **Web Interface:** Streamlit Framework
* **Data Extraction:** PyPDF2 Engine

## 💻 Local Setup & Installation

1. Clone this repository:
```bash
git clone [https://github.com/MeenaManisha-K/AI-Resume-Matcher.git](https://github.com/MeenaManisha-K/AI-Resume-Matcher.git)
cd AI-Resume-Matcher