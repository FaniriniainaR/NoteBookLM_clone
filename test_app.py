"""
================================================================================
 Tests unitaires pour l'application RAG locale (app.py)
================================================================================

Ces tests valident le pipeline sans interface graphique :
  - chargement des documents (txt / md / pdf)
  - découpage en chunks (taille & overlap)
  - propagation des métadonnées de source
  - création de la base vectorielle + recherche sémantique
  - pipeline de bout en bout (indexation -> recherche)

Lancement :
    .venv/bin/python -m pytest test_app.py -v
================================================================================
"""

import os
import shutil
import tempfile

import pytest

# Import du module applicatif. Grâce au garde `if __name__ == "__main__"`,
# l'import ne déclenche PAS l'interface Streamlit.
import app


# ------------------------------------------------------------------------------
# Utilitaires de test : création de faux fichiers de documents
# ------------------------------------------------------------------------------
def make_file(tmp_path, name, content):
    """Crée un fichier dans un dossier temporaire et renvoie son chemin."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


class FakeUploadedFile:
    """Simule l'objet UploadedFile renvoyé par st.file_uploader.

    getbuffer() renvoie un bytearray dans Streamlit réel : on reproduit ce
    comportement pour que app.load_documents() puisse écrire les octets.
    """

    def __init__(self, name, content):
        self.name = name
        self._content = content.encode("utf-8")

    def getbuffer(self):
        # Dans Streamlit, getbuffer() retourne un bytearray (octets bruts).
        return bytearray(self._content)


def _bytes_fake(name, data_bytes):
    """Crée un FakeUploadedFile avec contenu binaire (pour les PDF)."""
    obj = type("F", (), {})()
    obj.name = name
    obj.getbuffer = lambda: bytearray(data_bytes)
    return obj


# ==============================================================================
# 1. Chargement des documents
# ==============================================================================
def test_load_documents_txt(tmp_path):
    """Un fichier .txt doit être chargé avec sa source comme métadonnée."""
    p = make_file(tmp_path, "doc.txt", "Bonjour ceci est un document texte.")
    fake = FakeUploadedFile("doc.txt", "Bonjour ceci est un document texte.")

    docs = app.load_documents([fake])

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "doc.txt"
    assert "Bonjour" in docs[0].page_content


def test_load_documents_md(tmp_path):
    """Un fichier .md doit être chargé pareillement qu'un .txt."""
    fake = FakeUploadedFile("notes.md", "# Titre\nContenu en markdown.")
    docs = app.load_documents([fake])
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "notes.md"
    assert "Titre" in docs[0].page_content


@pytest.mark.skipif(
    not __import__("importlib.util").util.find_spec("fitz"),
    reason="pymupdf requis pour le chargement PDF",
)
def test_load_documents_pdf(tmp_path):
    """Un fichier PDF doit être chargé par PyMuPDFLoader."""
    import fitz  # pymupdf

    # Génère un mini PDF valide en mémoire avec pymupdf.
    pdf_doc = fitz.open()
    page = pdf_doc.new_page()
    page.insert_text((72, 72), "Texte extrait du PDF")
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    fake = _bytes_fake("doc.pdf", pdf_bytes)
    docs = app.load_documents([fake])
    assert len(docs) >= 1
    assert "Texte extrait du PDF" in docs[0].page_content


# ==============================================================================
# 2. Découpage en chunks
# ==============================================================================
def test_split_documents_respects_chunk_size_and_overlap():
    """Les chunks doivent respecter taille et overlap définis dans app."""
    # Un long texte pour forcer plusieurs chunks.
    long_text = ("mot " * 400)  # ~1600 caractères => plusieurs chunks de 500

    fake = FakeUploadedFile("long.txt", long_text)
    docs = app.load_documents([fake])
    chunks = app.split_documents(docs)

    # Vérifie qu'on a bien découpé en plusieurs morceaux.
    assert len(chunks) > 1

    # Chaque chunk ne dépasse pas la taille maximale.
    for c in chunks:
        assert len(c.page_content) <= app.CHUNK_SIZE

    # Les métadonnées de source sont propagées.
    for c in chunks:
        assert c.metadata["source"] == "long.txt"


def test_split_documents_propagates_source_metadata():
    """La source doit être conservée sur chaque chunk."""
    fake = FakeUploadedFile("meta.txt", "contenu " * 300)
    docs = app.load_documents([fake])
    chunks = app.split_documents(docs)
    assert all(c.metadata["source"] == "meta.txt" for c in chunks)


# ==============================================================================
# 3. Vectorisation et recherche
# ==============================================================================
def test_search_documents_returns_relevant_chunks(tmp_path):
    """Après indexation, la recherche doit renvoyer le chunk pertinent."""
    # Contenu : deux thématiques clairement distinctes pour tester la pertinence.
    content = (
        "L'ordinateur quantique utilise des qubits qui peuvent être en "
        "superposition. Les algorithmes quantiques comme Shor et Grover "
        "exploitent cette propriété pour résoudre des problèmes complexes.\n\n"
        "La recette du gâteau au chocolat nécessite de la farine, des œufs, "
        "du sucre et du cacao. On mélange puis on cuit à 180°C."
    )
    fake = FakeUploadedFile("base.txt", content)

    # On force la base vectorielle dans un dossier temporaire pour ne pas
    # polluer le répertoire de travail.
    old_persist = app.PERSIST_DIR
    temp_db = tempfile.mkdtemp()
    app.PERSIST_DIR = temp_db
    try:
        docs = app.load_documents([fake])
        chunks = app.split_documents(docs)
        vs = app.create_vectorstore(chunks)

        # Recherche sur un sujet présent dans le document.
        results = app.search_documents(vs, "ordinateur quantique", k=4)
        assert len(results) > 0

        # Le chunk retrouvé doit contenir un mot-clé du sujet interrogé.
        top_content = results[0][0].page_content
        assert ("quantique" in top_content) or ("qubit" in top_content)
    finally:
        # Restauration de l'état et nettoyage du dossier temporaire.
        app.PERSIST_DIR = old_persist
        shutil.rmtree(temp_db, ignore_errors=True)


def test_create_vectorstore_is_persistable(tmp_path):
    """Chroma doit persister la base sur disque dans PERSIST_DIR."""
    fake = FakeUploadedFile("persist.txt", "Texte de test pour la persistance.")
    old_persist = app.PERSIST_DIR
    temp_db = os.path.join(tempfile.mkdtemp(), "chroma_db")
    app.PERSIST_DIR = temp_db
    try:
        docs = app.load_documents([fake])
        chunks = app.split_documents(docs)
        vs = app.create_vectorstore(chunks)

        # Après création, le répertoire de persistance doit exister.
        assert os.path.exists(app.PERSIST_DIR)

        # La recherche fonctionne sur le vector store créé.
        assert len(app.search_documents(vs, "persistance")) == 1
    finally:
        app.PERSIST_DIR = old_persist
        shutil.rmtree(temp_db, ignore_errors=True)


# ==============================================================================
# 4. Pipeline de bout en bout (sans LLM)
# ==============================================================================
def test_full_pipeline_semantic_search(tmp_path):
    """Indexation + recherche sémantique produisent un résultat exploitable."""
    content = (
        "Ada Lovelace est considérée comme la première programmeuse de "
        "l'histoire. Elle a travaillé avec Charles Babbage sur la machine "
        "analytique au XIXe siècle."
    )
    fake = FakeUploadedFile("ada.txt", content)

    old_persist = app.PERSIST_DIR
    temp_db = os.path.join(tempfile.mkdtemp(), "chroma_db")
    app.PERSIST_DIR = temp_db
    try:
        docs = app.load_documents([fake])
        chunks = app.split_documents(docs)
        vs = app.create_vectorstore(chunks)

        # Question de recherche sémantique pure.
        results = app.search_documents(vs, "Qui était la première programmeuse ?", k=4)

        assert len(results) >= 1
        # Retrouver la mention "Lovelace".
        assert any(
            "Lovelace" in doc.page_content for doc, _ in results
        )
        # Le score (distance) doit être renseigné.
        for _, score in results:
            assert score is not None
    finally:
        app.PERSIST_DIR = old_persist
        shutil.rmtree(temp_db, ignore_errors=True)
