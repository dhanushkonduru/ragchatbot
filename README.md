# RAGChat

## 🧠 Chat with Your PDFs & Websites – RAG-Powered Multi-Source Q&A Bot
This project is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload multiple PDFs **or crawl websites** and ask natural language questions about their content. It combines semantic vector search with large language models (LLMs) to deliver accurate, context-rich answers.

The system leverages modern AI components such as vector databases, transformer-based embeddings, and LLMs to make document understanding interactive and scalable.

### 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   cd rag_chatbot
   pip install -r requirements.txt
   ```
   
   **Note:** If you plan to use website crawling, you'll also need to install Playwright browsers:
   ```bash
   playwright install chromium
   ```

2. **Set Up Environment Variables**
   
   Create a `.env` file in the `rag_chatbot` directory with the following:
   ```env
   # Groq API Key (required)
   # Get your API key from: https://console.groq.com/keys
   GROQ_API_KEY=your_groq_api_key_here
   
   # Qdrant Configuration
   # For local Qdrant: http://localhost:6333
   # For Qdrant Cloud: https://your-cluster-id.qdrant.io
   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=your_qdrant_api_key_here
   
   # Embedding Model (optional, defaults to all-MiniLM-L6-v2)
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   
   # Groq LLM Model (optional, defaults to llama-3.1-8b-instant)
   # Available models: llama-3.1-8b-instant, llama-3.3-70b-versatile, mixtral-8x7b-32768
   # The app will automatically try fallback models if the specified one is decommissioned
   GROQ_MODEL=llama-3.1-8b-instant
   ```

3. **Verify Setup (Optional but Recommended)**
   ```bash
   cd rag_chatbot
   python check_setup.py
   ```
   This will check if all dependencies are installed and configuration is correct.

4. **Run the Application**
   ```bash
   # Make sure you're in the rag_chatbot directory
   source ../venv/bin/activate  # or activate your virtual environment
   python -m streamlit run app.py
   ```

5. **Access the App**
   - Open your browser to `http://localhost:8501`
   - Upload PDFs or crawl websites and start asking questions!

### 📘 What is RAG?
RAG (Retrieval-Augmented Generation) is a powerful approach that enhances language models by feeding them relevant external context (like documents or PDFs) retrieved through vector search.

How it works:

Retrieval – Retrieve relevant chunks of data using semantic similarity.

Augmentation – Inject that context into the prompt for the language model.

Generation – Generate a human-like answer using the LLM based on the question + retrieved context.

This allows the model to give grounded, fact-based responses beyond its training knowledge.

### 📦 What is a Vector Database?
A Vector Database stores data as high-dimensional vectors instead of plain text. It enables semantic similarity search, allowing queries like:

"Find me the most relevant paragraphs across these PDFs for the question: 'How does insulin affect blood sugar?'"

In this project, we use Qdrant, a high-performance open-source vector database, to:

Store vector embeddings of PDF chunks

Efficiently search for top-matching document pieces during queries

### 🚀 How It Works

#### 1. Document Ingestion

**PDF Upload & Embedding:**
- Users upload PDF files via the Streamlit interface
- Each PDF is:
  - Loaded and split into small text chunks (1000 chars, 200 overlap)
  - Transformed into vector embeddings using SentenceTransformer
  - Indexed in Qdrant with metadata (source, text, source_type)

**Website Crawling & Embedding:**
- Users enter URLs (single or comma-separated) in the sidebar
- Set crawl depth (1-5 pages, default: 3)
- The system:
  - Checks robots.txt before crawling
  - Crawls pages with rate limiting (1 req/sec minimum)
  - Extracts clean text using trafilatura (removes boilerplate)
  - Falls back to Playwright for JavaScript-heavy sites
  - Splits text into 500-token chunks (≈2000 chars) with 50-token overlap
  - Deduplicates similar chunks (>95% similarity threshold)
  - Stores in Qdrant with metadata (source_url, page_title, crawl_date, domain, source_type)

#### 2. Querying with RAG
A user enters a question in natural language.

The system:
- Converts the question into an embedding
- Searches for the most semantically relevant text chunks from all selected documents (PDFs and websites)
- Aggregates the results and builds a prompt

#### 3. LLM-Powered Answering
The retrieved chunks + question are sent to Groq's LLaMA models

The LLM generates a detailed and accurate answer using both the prompt and embedded knowledge, following strict rules:
- Answers ONLY from provided context
- Cites sources with format: [Source: {title}]
- Lists all unique sources used

#### 4. Answer Display
The response is displayed in a clean UI with:
- Source type badges (📄 PDF or 🌐 Website)
- Clickable URLs in source citations
- Clear attribution of which documents contributed to the answer

Users can interact in real time and upload more documents or crawl additional websites

### 🧰 Tech Stack
Component	Technology
UI	Streamlit
Embedding Model	all-MiniLM-L6-v2 (via sentence-transformers)
Text Parsing	LangChain (PyPDFLoader, TextSplitter)
Web Scraping	BeautifulSoup4, trafilatura, Playwright (fallback)
Vector Database	Qdrant
Language Model	Groq API (LLaMA models)
Backend Logic	Python

### 🧩 Key Features
✅ Multi-PDF semantic search

✅ Website crawling with configurable depth

✅ Automatic PDF and web content embedding

✅ LLM-based natural language answers

✅ Source tracking with type badges (📄 PDF / 🌐 Website)

✅ Clickable URLs in source citations

✅ Robots.txt compliance and rate limiting

✅ Deduplication of similar content chunks

✅ Dark/light mode UI

✅ Streamlit-native file uploads and chat flow

### 🌐 Website Ingestion

#### How to Use:
1. **Enter URLs** in the sidebar under "🌐 Ingest Websites"
   - Single URL: `https://example.com`
   - Multiple URLs: `https://example.com, https://another-site.com`
   - Or separate by newlines

2. **Set Crawl Depth** (1-5 pages)
   - Depth 1: Only the URL you entered
   - Depth 3: URL + 2 levels of linked pages (default)
   - Depth 5: URL + 4 levels deep

3. **Click "🚀 Crawl & Embed URLs"**
   - Watch progress indicators as pages are crawled
   - Content is automatically extracted, chunked, and embedded
   - Collection name is auto-generated from domain

#### Features:
- **Robots.txt Compliance**: Automatically checks and respects robots.txt
- **Rate Limiting**: Minimum 1 second delay between requests per domain
- **Smart Extraction**: Uses trafilatura to remove boilerplate (nav, footer, ads)
- **JavaScript Support**: Falls back to Playwright for JS-heavy sites
- **Same-Domain Only**: Only crawls links within the same domain by default
- **Error Handling**: Gracefully handles timeouts, blocked sites, and inaccessible pages
- **Deduplication**: Removes chunks with >95% similarity

#### Best Practices:
- Start with depth 1-2 for testing
- Use depth 3-5 for comprehensive documentation sites
- Be respectful: the crawler respects robots.txt and uses rate limiting
- Some sites may block automated access - try different URLs if needed

#### Example Use Cases:
- 📚 Documentation sites (e.g., Python docs, API references)
- 📰 Blog posts and articles
- 📖 Wikipedia articles
- 🏢 Company websites and knowledge bases
