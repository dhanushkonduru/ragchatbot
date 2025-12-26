import os
from qdrant_client import QdrantClient

from utils.groq_client import get_groq_client
from utils.model_cache import get_embedding_model

EMBED_DIM = 384

def ask_pdf(question: str, collections: list, top_k=6) -> str:
    if not collections:
        raise ValueError("No collections provided for querying")
    
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    try:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        model = get_embedding_model()
        q_emb = model.encode(question)

        all_hits = []
        for collection in collections:
            try:
                # Use query_points instead of search (newer Qdrant API)
                response = qdrant.query_points(
                    collection_name=collection,
                    query=q_emb,  # Pass vector directly as query
                    limit=top_k
                )
                all_hits.extend(response.points)
            except Exception as e:
                raise RuntimeError(f"Error querying collection '{collection}': {str(e)}") from e

        if not all_hits:
            return "I couldn't find any relevant information in the selected PDFs to answer your question. Please try rephrasing your question or selecting different PDFs."

        # Sort by score and pick top overall
        sorted_hits = sorted(all_hits, key=lambda h: h.score, reverse=True)[:top_k]

        context = "\n---\n".join([f"[{h.payload.get('source', '')}] {h.payload.get('text', '')}" for h in sorted_hits])

        prompt = f"""You are an intelligent PDF assistant. Answer the following question using the information provided below.

Question: {question}

Relevant excerpts from the PDFs:
{context}

Provide a clear, complete and well-explained answer based on the information above. If multiple points are relevant, summarize them all. If the information doesn't directly answer the question, say so."""
        
        groq = get_groq_client()
        # Use environment variable for model, with fallback to current available models
        # Try models in order: user preference -> stable models
        user_model = os.getenv("GROQ_MODEL")
        fallback_models = [
            "llama-3.1-8b-instant",      # Fast, stable 8B model (most reliable)
            "llama-3.3-70b-versatile",   # Latest 70B model (if available)
            "mixtral-8x7b-32768",        # Alternative model
            "llama-3.1-70b-versatile"    # Previous version (fallback)
        ]
        
        model_name = user_model if user_model else fallback_models[0]
        
        # Try the model, with fallback if it's decommissioned
        try:
            resp = groq.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
        except Exception as e:
            error_str = str(e)
            # If model is decommissioned or invalid, try fallback models
            if ("decommissioned" in error_str.lower() or "not found" in error_str.lower() or "invalid" in error_str.lower()) and model_name not in fallback_models[1:]:
                for fallback_model in fallback_models[1:]:
                    try:
                        resp = groq.chat.completions.create(
                            model=fallback_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=1000
                        )
                        # Success! Use this model
                        model_name = fallback_model
                        break
                    except Exception as fallback_error:
                        # Try next fallback
                        continue
                else:
                    # All models failed, raise a helpful error
                    raise RuntimeError(
                        f"❌ All Groq models failed. The model '{model_name}' is not available.\n\n"
                        f"**Original error:** {error_str}\n\n"
                        f"**Solution:** Please set a valid GROQ_MODEL in your .env file.\n"
                        f"Check available models at: https://console.groq.com/docs/models\n"
                        f"Common working models: llama-3.1-8b-instant, llama-3.3-70b-versatile, mixtral-8x7b-32768"
                    ) from e
            else:
                # Re-raise if it's a different error or we've exhausted fallbacks
                raise

        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Error processing query: {str(e)}") from e
