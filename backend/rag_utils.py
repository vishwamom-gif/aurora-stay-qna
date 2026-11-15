
import re
from typing import Any

RETRIEVAL_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "can", "could", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "should", "the", "to", "was", "what", "when", "where", "who",
    "will", "with", "would", "you", "your", "hotel", "aurorastay"
}

def _metadata_property_codes(metadata: dict[str, Any]) -> set[str]:
    value = metadata.get("property_codes", [])
    if isinstance(value, str):
        values = re.split(r"[,;| ]+", value)
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = []
    return {str(item).strip().upper() for item in values if str(item).strip()}

def _query_terms(text: str) -> set[str]:
    terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {term for term in terms if term not in RETRIEVAL_STOPWORDS and len(term) > 1}

def retrieve(query: str, k: int = 2, vectorstore=None) -> list[str]:
    """
    Retrieve a wider semantic candidate pool, apply property-code eligibility
    when the query names a property, then rerank by query-term coverage.

    Generic policy chunks (without property_codes metadata) remain eligible
    because some policies apply across properties. This is a reranking heuristic,
    not a replacement for empirical retrieval evaluation.
    """
    if vectorstore is None:
        raise ValueError("Vector store is required.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    total_vectors = getattr(getattr(vectorstore, "index", None), "ntotal", 0)
    candidate_k = min(total_vectors or max(k * 8, 20), max(k * 8, 20))
    candidates = vectorstore.similarity_search(query, k=candidate_k)

    property_match = re.search(r"\bASH-[A-Z]{3}\b", query.upper())
    property_code = property_match.group(0) if property_match else None

    if property_code:
        eligible = []
        for doc in candidates:
            codes = _metadata_property_codes(doc.metadata or {})
            if not codes or property_code in codes:
                eligible.append(doc)
        candidates = eligible

    query_terms = _query_terms(query)

    def rerank_score(doc):
        content_terms = _query_terms(doc.page_content)
        coverage = (
            len(query_terms.intersection(content_terms)) / len(query_terms)
            if query_terms else 0.0
        )
        metadata_text = " ".join(
            str(value) for value in (doc.metadata or {}).values()
        ).lower()
        metadata_hits = sum(1 for term in query_terms if term in metadata_text)
        metadata_score = min(metadata_hits / max(len(query_terms), 1), 1.0)
        property_bonus = 1.0 if (
            property_code and property_code in _metadata_property_codes(doc.metadata or {})
        ) else 0.0
        return (0.75 * coverage) + (0.15 * metadata_score) + (0.10 * property_bonus)

    ranked = sorted(candidates, key=rerank_score, reverse=True)[:k]

    return [
        f"Content:\n{doc.page_content}\n\nMetadata:\n{doc.metadata}"
        for doc in ranked
    ]


from langchain_openai import ChatOpenAI
import os

def generate(query, retrieved_chunks, model) -> str:
    prompt = f"""
You are AuroraStay Hotels' internal policy assistant helping hotel staff answer guest questions.

User Query:
{query}

Retrieved Policy Information:
{retrieved_chunks}

Rules:
1. Use only information explicitly supported by the retrieved policy information.
2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
3. Include all relevant information needed to answer the query completely.
4. Do not omit important conditions, exclusions, limitations, exceptions, or coverage details.
5. Focus only on information directly relevant to the user's query.
6. Do not include unrelated information from the retrieved policy information.
7. Answer the user's specific question directly.
8. Keep the answer concise, clear, and easy to understand.
9. Do not invent room features, occupancy limits, fees, cancellation deadlines, refund terms, amenities, or property-specific rules.
10. If the information is insufficient, say:
"The available policy documents do not contain enough information to answer this question."
If retrieved information conflicts, clearly mention the conflict.

Answer:
"""
    return model.invoke(prompt)

def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE")
    )
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)
    answer = generate(query=query, retrieved_chunks=retrieved_chunks, model=model)
    return answer.content, retrieved_chunks
