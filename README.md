# 🎓 AI Study Buddy

An intelligent, AI-powered study assistant built with a fine-tuned Qwen 2.5 3B model, RAG pipeline, and an agentic student dashboard. Designed to help students learn smarter through personalized flashcards, quizzes, summaries, mind maps, and AI-driven feedback.

---

## ✨ Features

- **AI Chatbot** — Conversational study assistant powered by a fine-tuned Qwen 2.5 3B Instruct model with full session memory
- **Summarizer** — Paste notes or upload documents and get concise, topic-aware summaries
- **Flashcard Generator** — Automatically generates Q&A pairs from study material
- **Quiz Generator** — Creates multiple-choice questions from any topic or uploaded content
- **Mind Map** — Converts notes into a structured tree for visual learning
- **Student Dashboard** — Tracks performance per topic and surfaces personalized study suggestions using Agentic AI

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| Fine-tuned model | Qwen 2.5 3B Instruct (QLoRA 4-bit via Unsloth) |
| Model hosting | Hugging Face Hub |
| Backend | FastAPI (Python) |
| RAG | ChromaDB + Sentence Transformers |
| Student DB | SQLite / PostgreSQL via SQLAlchemy |
| Frontend | React |
| Inference | Transformers + BitsAndBytes (4-bit) |

---

## 🗺️ Project Roadmap

### Phase 1 — Model Serving
- Set up FastAPI with `/chat` and `/summarize` endpoints
- Load fine-tuned Qwen model from Hugging Face using 4-bit inference
- Connect frontend to backend (CORS, fetch/axios)

### Phase 2 — RAG Pipeline + Core Features
- Set up ChromaDB collections and document persistence
- Chunk, embed, and store study documents using Sentence Transformers
- Retrieve top-k relevant chunks and inject into Qwen prompt context
- Build Summarizer, Flashcard Generator, Quiz Generator, and Mind Map features on top of the RAG pipeline

### Phase 3 — Student Data + Dashboard UI
- Store quiz scores per topic per student in a database
- Build a React dashboard with performance charts and history
- Track weak topics based on score trends over time

### Phase 4 — Agentic AI
- Build a performance agent that reads scores and detects weak topics
- Create a suggestion engine that recommends summaries, quizzes, or flashcards
- Implement tool calling so the agent can invoke RAG and quiz tools dynamically
- Close the feedback loop: observe → reason → act → learn from user responses

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- A GPU or Hugging Face Inference API access
- A Hugging Face account with the fine-tuned model pushed

### Backend Setup

```bash
# Clone the repo
git clone https://github.com/your-username/ai-study-buddy.git
cd ai-study-buddy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn transformers bitsandbytes accelerate
pip install chromadb sentence-transformers sqlalchemy
```

### Run the Backend

```bash
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## 🔧 Model Details

- **Base model:** Qwen 2.5 3B Instruct
- **Fine-tuning method:** QLoRA 4-bit using [Unsloth](https://github.com/unslothai/unsloth)
- **Hosted on:** Hugging Face — `your-username/your-model-name`
- **Inference:** Loaded via `transformers` with `BitsAndBytesConfig` for 4-bit quantization

---

## 🗂️ Project Structure

```
ai-study-buddy/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── model.py             # Qwen model loading + inference
│   ├── rag.py               # ChromaDB setup + retrieval
│   ├── agent.py             # Agentic dashboard logic
│   └── database.py          # Student score tracking
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   └── pages/           # Dashboard, Chat, Quiz, etc.
│   └── package.json
├── data/
│   └── chroma_store/        # Persisted ChromaDB collections
└── README.md
```

---

## 🤖 Agentic Dashboard Logic

The student dashboard uses an agent loop to give personalized feedback:

1. **Observe** — reads the student's quiz scores per topic
2. **Reason** — identifies weak topics (e.g., score < 60%)
3. **Act** — suggests an action ("You scored 40% on Thermodynamics — want a detailed summary?")
4. **Learn** — tracks whether the student accepts the suggestion and adjusts future recommendations

The agent uses tool calling to dynamically invoke the right feature (quiz, summary, flashcards) based on the student's current state.

---

## 📦 Dependencies

```
fastapi
uvicorn
transformers
bitsandbytes
accelerate
chromadb
sentence-transformers
sqlalchemy
torch
unsloth (fine-tuning only)
```

---

## 🙌 Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) for efficient QLoRA fine-tuning
- [Qwen](https://huggingface.co/Qwen) by Alibaba Cloud for the base model
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [Hugging Face](https://huggingface.co/) for model hosting and the Transformers library

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.
