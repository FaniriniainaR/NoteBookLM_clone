"""
================================================================================
 RAG local — Assistant de recherche sémantique et génération contrainte
================================================================================

Cette application Streamlit met en œuvre un système RAG (Retrieval-Augmented
Generation) 100% local :

  - Aucun appel à une API externe (pas d'OpenAI, pas de clé cloud).
  - Les embeddings sont produits localement avec des modèles `sentence-transformers`.
  - Les vecteurs sont stockés localement dans une base ChromaDB persistante.
  - La génération LLM est assurée par un modèle exécuté via Ollama (localement).

Deux modes de fonctionnement sont disponibles :
  1. "Recherche sémantique pure"  -> on récupère et affiche les chunks pertinents,
                                     SANS génération LLM.
  2. "Assistant RAG complet"      -> on récupère les chunks puis on demande à un
                                     LLM local de générer une réponse contrainte
                                     par le contexte.

Par défaut, le mode LLM est DÉSACTIVÉ : on reste donc en recherche sémantique
pure tant que l'utilisateur n'a pas activé le toggle.

Usage :
    streamlit run app.py
================================================================================
"""

# ------------------------------------------------------------------------------
# Import des bibliothèques
# ------------------------------------------------------------------------------
import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

# --- LangChain (pipeline RAG) ---
# PyMuPDFLoader : extrait le texte de fichiers PDF via la bibliothèque PyMuPDF.
# TextLoader    : charge un fichier texte brut (.txt, .md).
# RecursiveCharacterTextSplitter : découpe le texte en "chunks" de taille fixe
#                                  en privilégiant les séparateurs naturels
#                                  (paragraphes, phrases, mots).
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
try:
    # LangChain >= 0.1 : text splitters déplacés dans un paquet dédié.
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Anciennes versions de LangChain : import depuis langchain directement.
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# HuggingFaceEmbeddings : produit des représentations vectorielles (embeddings)
# des chunks à partir d'un modèle de sentence-transformers, en local.
from langchain_community.embeddings import HuggingFaceEmbeddings

# Chroma : base vectorielle persistante locale. On stocke les vecteurs dans le
# dossier ./chroma_db pour ne pas les recalculer à chaque lancement.
from langchain_community.vectorstores import Chroma

# PromptTemplate : structure le prompt système/assistant de façon typée.
try:
    # LangChain récent : PromptTemplate vit dans langchain-core.
    from langchain_core.prompts import PromptTemplate
except ImportError:
    # Anciennes versions : dans langchain.prompts.
    from langchain.prompts import PromptTemplate

# ChatOllama : client pour interroger un modèle de génération exécuté par Ollama.
try:
    # LangChain récent : package d'intégration dédié à Ollama.
    from langchain_ollama import ChatOllama
except ImportError:
    # Anciennes versions : ChatOllama était dans langchain-community.
    from langchain_community.chat_models import ChatOllama

# ------------------------------------------------------------------------------
# Constantes de configuration
# ------------------------------------------------------------------------------
# Chemin du dossier de persistance de la base ChromaDB.
PERSIST_DIR = "./chroma_db"

# Nom fixe de la collection (on la vide plutôt que de la supprimer pour la
# réinitialiser à chaque nouvelle indexation).
COLLECTION_NAME = "documents"

# ----------------------- Paramètres de découpage (chunking) -------------------
# La taille de chunk (500 caractères) est un compromis :
#   - Trop petit (< 200)  -> perte de contexte, chunks fragmentés et peu
#                            exploitables par le LLM.
#   - Trop grand (> 1000) -> le recouvrement sémantique devient flou et la
#                            recherche moins précise ; on risque aussi de
#                            dépasser la fenêtre de contexte du LLM local.
# On choisit 500 caractères pour obtenir des unités de sens suffisamment riches
# tout en restant rapides à vectoriser et dans le contexte d'un LLM.
CHUNK_SIZE = 500

# L'overlap (50 caractères, soit 10% du chunk) permet de préserver le contexte
# entre deux chunks consécutifs : une phrase ou une idée qui chevauche la frontière
# d'un chunk ne sera pas perdue lors de la recherche.
CHUNK_OVERLAP = 50

# ----------------------- Modèle d'embedding ------------------------------------
# all-MiniLM-L6-v2 : petit, rapide et de très bon rapport qualité/taille (384 dim).
# Très adapté pour un usage local sans GPU. On pourrait utiliser all-mpnet-base-v2
# (768 dim, plus lourd mais plus précis) pour de meilleurs résultats.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------- Modèle LLM local (Ollama) -----------------------------
# Modèle utilisé par le mode RAG complet. mistral est un bon choix générique ;
# qwen2.5-coder conviendrait mieux pour du code. Il doit avoir été téléchargé
# au préalable (ollama pull mistral).
OLLAMA_MODEL = "mistral"

# Adresse du serveur Ollama (par défaut localhost:11434).
OLLAMA_BASE_URL = "http://localhost:11434"

# Nombre de chunks récupérés lors de la recherche (k voisins les plus proches).
TOP_K = 4

# Température du LLM : 0.2 => faible créativité, on privilégie la fidélité au
# contexte (important pour un système RAG où l'on veut éviter l'hallucination).
LLM_TEMPERATURE = 0.2


# ==============================================================================
# Fonctions du pipeline d'ingestion
# ==============================================================================

def load_documents(uploaded_files):
    """
    Charge en mémoire les documents téléversés par l'utilisateur.

    Paramètres
    ----------
    uploaded_files : list
        Liste d'objets `UploadedFile` provenant de `st.file_uploader`.

    Retour
    ------
    list[Document]
        Liste de documents LangChain (contenant .page_content et .metadata),
        chaque document portant sa métadonnée "source" (nom du fichier).
    """
    documents = []

    # On écrit chaque fichier téléversé dans un dossier temporaire : les loaders
    # LangChain (PyMuPDFLoader, TextLoader) travaillent à partir de chemins de
    # fichiers, pas d'objets en mémoire.
    with tempfile.TemporaryDirectory() as tmpdir:
        for uploaded_file in uploaded_files:
            # Nom d'origine du fichier (servira de métadonnée "source").
            filename = uploaded_file.name

            # Déterminer l'extension en minuscules pour choisir le bon loader.
            ext = Path(filename).suffix.lower()

            # Enregistrer le fichier sur disque (dans le dossier temporaire).
            tmp_path = os.path.join(tmpdir, filename)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if ext == ".pdf":
                # PyMuPDFLoader extrait le texte de chaque page du PDF.
                loader = PyMuPDFLoader(tmp_path)
            elif ext in (".txt", ".md"):
                # TextLoader lit un fichier texte brut (.txt et .md).
                loader = TextLoader(tmp_path, encoding="utf-8")
            else:
                # Format non supporté : on l'ignore gracieusement avec un message.
                st.warning(f"Format non supporté, fichier ignoré : {filename}")
                continue

            # Le loader renvoie UN document par page/fichier. Chacun porte déjà
            # une métadonnée "source" = chemin complet. On la remplace par le nom
            # du fichier seul pour un affichage propre.
            for doc in loader.load():
                doc.metadata["source"] = filename
                documents.append(doc)

    return documents


def split_documents(documents):
    """
    Découpe les documents en chunks de taille homogène.

    Paramètres
    ----------
    documents : list[Document]
        Liste de documents chargés par `load_documents`.

    Retour
    ------
    list[Document]
        Liste de chunks, chacun avec sa métadonnée "source" héritée du document
        parent (utile pour retracer l'origine d'un extrait).
    """
    # RecursiveCharacterTextSplitter découpe d'abord sur les "gros" séparateurs
    # (\n\n, \n, ponctuation, espace...) puis raffine : on obtient ainsi des
    # chunks "naturels" plutôt qu'une coupure brutale en plein milieu d'un mot.
    # Voir les constantes CHUNK_SIZE / CHUNK_OVERLAP commentées plus haut.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,            # la "longueur" = nombre de caractères
        add_start_index=True,           # ajoute start_index (position dans le doc)
    )

    # split_documents conserve automatiquement .metadata de chaque document et le
    # propage à chaque chunk généré (donc "source" est préservé).
    return text_splitter.split_documents(documents)


def create_vectorstore(documents):
    """
    Crée (ou recrée) la base vectorielle Chroma persistante à partir des chunks.

    Paramètres
    ----------
    documents : list[Document]
        Les chunks à indexer, produits par `split_documents`.

    Retour
    ------
    Chroma
        Un vector store prêt à être interrogé.

    Remarque
    --------
    La collection est VIDEÉ à chaque appel : cela garantit qu'une nouvelle
    indexation écrase l'ancienne et évite l'accumulation de doublons.
    """
    # On libère d'abord l'éventuel vector store en cache : sa connexion SQLite
    # reste ouverte sur PERSIST_DIR, et écraser le dossier alors qu'elle est
    # active provoque l'erreur "attempt to write a readonly database".
    get_chroma_vectorstore.clear()

    # On recrée les embeddings à chaque indexation via le cache_resource
    # (voir get_embeddings) pour éviter de recharger le modèle inutilement.
    embeddings = get_embeddings()

    # Si la collection existe déjà sur le disque, on détruit son contenu pour
    # repartir "de zéro" (réinitialisation demandée par le sujet).
    if os.path.exists(PERSIST_DIR):
        # On ouvre la collection existante par-dessus pour la VIDER via l'API,
        # plutôt que de supprimer le dossier : ceci écrase proprement l'ancienne
        # base sans casser la connexion SQLite sous-jacente.
        try:
            old_vs = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
            )
            old_vs.delete_collection()
        except Exception:
            # En dernier recours, si la base est corrompue ou inutilisable,
            # on supprime physiquement le dossier pour repartir de zéro.
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    # Chroma.from_documents : vectorise les chunks et les stocke de façon
    # persistante dans PERSIST_DIR. On fixe un nom de collection unique.
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


@st.cache_resource
def get_embeddings():
    """
    Fabrique (et met en cache) le modèle d'embedding local.

    Retour
    ------
    HuggingFaceEmbeddings
        Instance partagée du modèle d'embedding. La mise en cache via
        `@st.cache_resource` évite de recharger le modèle à chaque interaction,
        ce qui serait très coûteux en temps.
    """
    # Le paramètre model_name sélectionne le modèle sentence-transformers à
    # charger. Il est téléchargé automatiquement au premier usage (cache HF).
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def get_chroma_vectorstore():
    """
    Récupère la base vectorielle persistante déjà indexée (mode lecture).

    Retour
    ------
    Chroma | None
        Le vector store si la base existe sur disque, sinon None.
    """
    if not os.path.exists(PERSIST_DIR):
        return None
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


# ==============================================================================
# Fonctions de recherche et de génération
# ==============================================================================

def search_documents(vectorstore, query, k=TOP_K):
    """
    Recherche les k chunks les plus proches sémantiquement de la question.

    Paramètres
    ----------
    vectorstore : Chroma
        La base vectorielle à interroger.
    query : str
        La question posée par l'utilisateur.
    k : int
        Nombre de chunks à retourner (par défaut TOP_K = 4).

    Retour
    ------
    list[tuple[Document, float]]
        Liste de tuples (document, score de similarité) triés par pertinence.
    """
    # similarity_search_with_score renvoie les k plus proches voisins : on
    # transforme la question en vecteur puis on calcule la distance à chaque
    # chunk stocké. Le score est une distance (plus petit = plus proche).
    return vectorstore.similarity_search_with_score(query, k=k)


def generate_response(question, retrieved_docs):
    """
    Génère une réponse via un LLM local (Ollama), contrainte par le contexte.

    Paramètres
    ----------
    question : str
        La question posée par l'utilisateur.
    retrieved_docs : list[tuple[Document, float]]
        Les chunks récupérés (même format que le retour de search_documents).

    Retour
    ------
    str
        La réponse générée par le LLM.
    """
    # --- 1. Assemblage du contexte à partir des chunks récupérés ---
    # On concatène le contenu des chunks, séparés par des sauts de ligne, pour
    # former le bloc "contexte" qui sera injecté dans le prompt.
    context = "\n\n".join(doc.page_content for doc, _ in retrieved_docs)

    # --- 2. Construction du prompt avec PromptTemplate ---
    # Ce template impose un rôle strict à l'assistant et l'oblige à ne répondre
    # qu'à partir du contexte fourni : c'est la clé pour limiter les
    # hallucinations et garantir des réponses ancrées dans les documents.
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Vous êtes un assistant utile. Répondez à la question de l'utilisateur "
            "en vous basant UNIQUEMENT sur le contexte fourni. "
            "Si la réponse n'est pas dans le contexte, dites exactement : "
            '"Je ne trouve pas cette information dans les documents fournis."\n\n'
            "Contexte :\n{context}\n\n"
            "Question : {question}\n\n"
            "Réponse :"
        ),
    )

    # On instancie le chat model Ollama. La température basse (0.2) réduit la
    # créativité pour rester fidèle au contexte. Le modèle doit être disponible
    # localement (voir la vérification dans main()).
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
    )

    # --- 3. Formation de la chaîne prompt -> LLM et invocation ---
    # .format_prompt(...) remplit le template, puis .invoke() envoie la requête
    # au modèle et récupère la réponse.
    chain = prompt_template | llm
    response = chain.invoke({"context": context, "question": question})

    # ChatOllama renvoie un objet AIMessage ; on retourne son contenu textuel.
    return response.content


def check_ollama_availability():
    """
    Vérifie si le serveur Ollama répond (le cas échéant, remonte un warning).

    Retour
    ------
    bool
        True si Ollama est joignable, False sinon.
    """
    try:
        # Une simple requête sur la liste des modèles permet de tester la
        # disponibilité du serveur sans rien modifier.
        import urllib.request
        import json

        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # Ollama renvoie les modèles avec leur tag (ex: "mistral:latest").
            # On retire le suffixe ":tag" pour comparer au nom de base ("mistral").
            models = [m["name"].split(":")[0] for m in data.get("models", [])]
            return OLLAMA_MODEL in models, models
    except Exception:
        return False, []


# ==============================================================================
# Fonction principale (interface Streamlit)
# ==============================================================================

def main():
    """
    Point d'entrée de l'application Streamlit : met en place la sidebar,
    gère l'indexation et la boucle de conversation.
    """
    # --- Configuration globale de la page (doit être la première commande) ---
    st.set_page_config(page_title="RAG local", page_icon="📚", layout="wide")

    st.title("📚 Assistant RAG 100% local")
    st.caption(
        "Chargement de documents, recherche sémantique et génération assistée "
        "par un LLM local (Ollama). Aucune donnée ne quitte votre machine."
    )

    # =========================================================================
    # BARRE LATÉRALE : téléversement, indexation, options
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration")

        # --- Zone de téléversement de fichiers (multi-fichiers autorisé) ---
        uploaded_files = st.file_uploader(
            "Téléverser vos documents (.pdf, .md, .txt)",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True,
        )

        # --- Bouton d'indexation ---
        if st.button("📥 Indexer les documents", use_container_width=True):
            if not uploaded_files:
                # Rien à indexer : on prévient l'utilisateur.
                st.warning("Veuillez d'abord téléverser au moins un fichier.")
            else:
                # Spinner pour signaler que l'indexation est en cours.
                with st.spinner("Indexation en cours..."):
                    docs = load_documents(uploaded_files)      # extraction
                    chunks = split_documents(docs)             # découpage
                    create_vectorstore(chunks)                 # vectorisation
                    # Vide le cache du vector store pour relire la nouvelle base.
                    get_chroma_vectorstore.clear()
                st.success(f"{len(chunks)} chunks indexés.")

        # --- État de la base vectorielle ---
        st.divider()
        st.subheader("📊 Base vectorielle")

        # On interroge la base persistante pour afficher le nombre de chunks.
        vectorstore = get_chroma_vectorstore()
        if vectorstore is None:
            st.info("Aucun document indexé pour le moment. Utilisez le bouton "
                    "« Indexer les documents ».")
        else:
            # Chroma expose la collection ; .count() donne le nombre d'items.
            try:
                nb_chunks = vectorstore._collection.count()
            except Exception:
                nb_chunks = 0
            st.write(f"**{nb_chunks}** chunks indexés dans `{COLLECTION_NAME}`.")

        st.divider()

        # --- Toggle mode LLM (RAG complet) ---
        # Par défaut désactivé => on est en "Recherche sémantique pure".
        use_llm = st.toggle(
            "🤖 Mode RAG complet (LLM)",
            value=False,
            help="Activez pour utiliser le LLM local Ollama. Sinon, seule la "
                 "recherche sémantique est effectuée.",
        )

        if use_llm:
            # On vérifie la disponibilité d'Ollama pour prévenir tôt.
            available, models = check_ollama_availability()
            if not available:
                st.warning(
                    f"⚠️ Ollama semble indisponible ou le modèle « {OLLAMA_MODEL} » "
                    "n'est pas installé. Vérifiez `ollama pull " + OLLAMA_MODEL + "` "
                    "et que le serveur tourne."
                )
            else:
                st.success(f"Ollama prêt (modèle « {OLLAMA_MODEL} »).")

    # =========================================================================
    # ZONE PRINCIPALE : historique conversationnel
    # =========================================================================
    # On initialise l'historique des messages si ce n'est pas déjà fait
    # (les objets de session sont persistés entre les reruns de Streamlit).
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage de l'historique existant.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # On réaffiche les éventuelles sources associées à une réponse.
            if "sources" in message and message["sources"]:
                with st.expander("Voir les sources"):
                    for doc, score in message["sources"]:
                        st.markdown(f"**Source :** `{doc.metadata.get('source', '?')}`")
                        st.markdown(f"*Extrait :*\n{doc.page_content}")
                        st.divider()

    # =========================================================================
    # INPUT UTILISATEUR
    # =========================================================================
    prompt = st.chat_input("Posez votre question sur les documents...")

    if prompt:
        # --- Ajout du message utilisateur à l'historique et affichage ---
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Vérification préalable : faut-il avoir indexé des documents ? ---
        if vectorstore is None:
            # Aucune base indexée : on répond directement avec un message d'alerte.
            with st.chat_message("assistant"):
                st.warning(
                    "Aucun document n'a encore été indexé. "
                    "Utilisez la barre latérale pour téléverser et indexer vos fichiers."
                )
            st.session_state.messages.append(
                {"role": "assistant", "content": "Aucun document indexé.", "sources": []}
            )
            st.stop()

        # --- Recherche sémantique (commune aux deux modes) ---
        with st.spinner("Recherche des passages pertinents..."):
            results = search_documents(vectorstore, prompt, k=TOP_K)

        # ---------------------------------------------------------------------
        # MODE 1 : Recherche sémantique pure (toggle LLM désactivé)
        # ---------------------------------------------------------------------
        if not use_llm:
            if not results:
                with st.chat_message("assistant"):
                    st.markdown("Aucun extrait trouvé.")
            else:
                # On affiche "brutalement" le contenu de chaque chunk avec sa
                # source, dans la zone conversationnelle.
                with st.chat_message("assistant"):
                    st.markdown("#### 🔍 Résultats de la recherche sémantique")
                    for doc, score in results:
                        st.markdown(
                            f"**Source :** `{doc.metadata.get('source', 'inconnue')}`  ·  "
                            f"*similarité : {score:.3f}*"
                        )
                        st.markdown(f"**Extrait :**\n\n{doc.page_content}")
                        st.divider()

                # On conserve l'affichage dans l'historique.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Aperçu des extraits pertinents (recherche sémantique).",
                        "sources": results,
                    }
                )

        # ---------------------------------------------------------------------
        # MODE 2 : Assistant RAG complet (toggle LLM activé)
        # ---------------------------------------------------------------------
        else:
            if not results:
                with st.chat_message("assistant"):
                    st.markdown("Je ne trouve pas cette information dans les documents fournis.")
            else:
                # Génération de la réponse à partir du contexte récupéré.
                with st.spinner("Génération de la réponse (LLM local)..."):
                    try:
                        answer = generate_response(prompt, results)
                    except Exception as e:
                        # Si Ollama ne répond pas, on ne fait pas planter l'app.
                        answer = (
                            "⚠️ Erreur lors de l'appel au LLM local : "
                            f"`{e}`\n\nVérifiez qu'Ollama est lancé et que le "
                            f"modèle « {OLLAMA_MODEL} » est installé."
                        )

                # Affichage de la réponse de l'assistant dans la conversation.
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    # Transparence : sous la réponse, un expander expose les
                    # chunks utilisés pour produire cette réponse.
                    with st.expander("Voir les sources"):
                        for doc, score in results:
                            st.markdown(
                                f"**Source :** `{doc.metadata.get('source', '?')}`  ·  "
                                f"*similarité : {score:.3f}*"
                            )
                            st.markdown(f"*Extrait :*\n{doc.page_content}")
                            st.divider()

                # On conserve la réponse et ses sources dans l'historique.
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": results}
                )


# ==============================================================================
# Point d'entrée du script
# ==============================================================================
if __name__ == "__main__":
    main()
