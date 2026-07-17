"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        # If docs_folder is a relative path like "docs", resolve it relative to
        # THIS file's location rather than the terminal's current working
        # directory. Otherwise running `python main.py` from a parent folder
        # would look for docs/ in the wrong place, load zero documents, and make
        # retrieval silently return "I do not know" for every query.
        if not os.path.isabs(docs_folder):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            docs_folder = os.path.join(script_dir, docs_folder)

        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Tokenizer (shared helper used by indexing, scoring, retrieval)
    # -----------------------------------------------------------

    def _tokenize(self, text):
        """
        Turn a chunk of text into a clean list of lowercase words.

        We do this in one place so that indexing, scoring, and retrieval all
        agree on what counts as a "word". If they disagreed, "token." in a doc
        might not match "token" in a query and retrieval would silently miss
        hits.

        Steps:
        1. Lowercase everything so "Token" and "token" are treated the same.
        2. Replace every run of non-alphanumeric characters (spaces, periods,
           slashes, angle brackets, etc.) with a single space. This strips
           punctuation like "database?" -> "database" and splits "/api/login"
           into "api" and "login".
        3. Split on whitespace into individual words.
        """
        text = text.lower()
        # \w matches letters, digits, and underscore; [^\w] is everything else.
        # Turning punctuation into spaces lets str.split() do the rest.
        text = re.sub(r"[^\w]+", " ", text)
        return text.split()

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Build a tiny inverted index: a dict mapping each lowercase word to the
        list of filenames that contain it.

        Example structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }

        An "inverted" index flips the natural direction (doc -> words) into
        (word -> docs), which is how real search engines answer "which
        documents mention X?" quickly. For this small activity the scoring
        below doesn't strictly need the index, but building it is the point of
        the exercise.
        """
        index = {}

        # documents is a list of (filename, text) tuples produced by
        # load_documents().
        for filename, text in documents:
            # Use a set so each word in this doc is counted once. Without this,
            # a word appearing 5 times would append the filename 5 times.
            words_in_doc = set(self._tokenize(text))

            for word in words_in_doc:
                # dict.setdefault gives us the existing list for this word, or
                # creates a new empty list the first time we see the word.
                index.setdefault(word, []).append(filename)

        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a simple relevance score for how well `text` matches `query`.

        Baseline used here: count how many distinct query words appear anywhere
        in the document. More overlapping words -> higher score.
        """
        # set(...) gives us the UNIQUE words in the query. We only care whether
        # a query word appears in the doc, not how many times it was typed.
        query_words = set(self._tokenize(query))
        text_words = set(self._tokenize(text))

        # The & operator is set intersection: the words present in BOTH the
        # query and the document. len(...) counts them -> that's the score.
        return len(query_words & text_words)

        # Tinker ideas (things you could try to improve ranking):
        # - Count total occurrences instead of presence (weights repeated words).
        # - Remove stopwords like "the", "do", "i", "to" so common filler words
        #   don't inflate the score. Right now "How do I connect to the database"
        #   gives free points to any doc containing "to" or "the".
        # - Divide by document length so short, focused docs aren't penalized.

    def retrieve(self, query, top_k=3):
        """
        Select the top_k most relevant documents for `query`.

        Returns a list of (filename, text) tuples sorted by score, best first.
        This is the method the rest of the app relies on (answer_retrieval_only
        and answer_rag both call it), so its output shape matters.
        """
        scored = []

        # Score every document we loaded. self.documents is the list of
        # (filename, text) tuples built in __init__.
        for filename, text in self.documents:
            score = self.score_document(query, text)

            # Skip documents with zero matching words. Keeping them would let
            # irrelevant files leak into the answer; dropping them lets the bot
            # honestly say "I do not know" when nothing matches, which is what
            # the RAG refusal prompt in llm_client.py expects.
            if score > 0:
                scored.append((score, filename, text))

        # Sort by score descending. key=lambda item: item[0] sorts on the score
        # (the first element of each tuple); reverse=True puts the highest first.
        scored.sort(key=lambda item: item[0], reverse=True)

        # Rebuild the (filename, text) shape the callers expect, dropping the
        # score, and keep only the best top_k.
        results = [(filename, text) for _, filename, text in scored]
        return results[:top_k]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
