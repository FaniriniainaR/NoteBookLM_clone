# NoteBookLM_clone — Assistant RAG 100% local

Application Streamlit implémentant un système **RAG** (Retrieval-Augmented Generation)
entièrement **local**, sans aucun appel à une API externe. L'utilisateur charge des
documents (PDF, Markdown, TXT) et interagit avec eux via deux modes.

---

## Fonctionnalités

- **Chargement de documents** : PDF, Markdown (`.md`), Texte (`.txt`) — multi-fichiers.
- **Deux modes d'interaction** :
  - 🔍 **Recherche sémantique pure** : récupère et affiche les chunks pertinents, sans LLM.
  - 🤖 **Assistant RAG complet** : génère une réponse via un LLM local (Ollama), contrainte par le contexte.
- **100% local** : vecteurs stockés dans ChromaDB (`./chroma_db`), embeddings par `sentence-transformers`, aucune donnée ne quitte la machine.
- **Transparence** : dans le mode RAG, un expander « Voir les sources » expose les extraits utilisés.

---

## Architecture

```
Documents (PDF/MD/TXT)
        │
        ▼
  Extraction (PyMuPDFLoader / TextLoader)
        │
        ▼
  Découpage (RecursiveCharacterTextSplitter, 500 / overlap 50)
        │
        ▼
  Embeddings (all-MiniLM-L6-v2, sentence-transformers)
        │
        ▼
  Stockage (ChromaDB, persistance locale ./chroma_db)
        │
        ▼
   Recherche (similarité, k=4)
        │
        ├── Mode sémantique ──► affichage brut des chunks
        └── Mode RAG ─────────► prompt strict → LLM local (Ollama) → réponse + sources
```

### Choix techniques documentés dans le code

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Taille de chunk | `500` caractères | Équilibre entre granularité de recherche et contexte exploitable par le LLM |
| Overlap | `50` caractères (10%) | Préserve le contexte entre chunks consécutifs |
| Modèle d'embedding | `all-MiniLM-L6-v2` | Rapide, léger (384 dim), idéal en local sans GPU |
| Modèle LLM | `mistral` (Ollama) | Bon rapport qualité/usage, exécuté localement |
| Température | `0.2` | Faible créativité → fidélité au contexte, limite les hallucinations |
| Top-k | `4` | Nombre de chunks injectés dans le contexte |

---

## Installation

```bash
# 1. Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Installer et démarrer Ollama, puis télécharger le modèle
#    (https://ollama.com)
ollama pull mistral
```

---

## Utilisation

```bash
streamlit run app.py
```

Puis ouvrez `http://localhost:8501`.

1. **Barre latérale** : téléversez vos fichiers (`.pdf`, `.md`, `.txt`), cliquez sur
   **« 📥 Indexer les documents »** (écrase l'ancienne base).
2. **Mode sémantique** (toggle désactivé par défaut) : posez une question → les
   extraits pertinents et leurs sources sont affichés.
3. **Mode RAG** (activez le toggle **« 🤖 Mode RAG complet (LLM) »**) : posez une
   question → le LLM local rédige une réponse basée uniquement sur le contexte,
   avec l'expander « Voir les sources ».

---

## Tests unitaires

Des tests couvrent le pipeline sans interface (chargement, découpage, vectorisation,
recherche, parcours complet).

```bash
python -m pytest test_app.py -v
```

---

## Structure du projet

```
app.py            # Application Streamlit complète (interface + pipeline RAG)
test_app.py       # Tests unitaires du pipeline
requirements.txt  # Dépendances Python
chroma_db/        # Base vectorielle persistante (générée, non versionnée)
```

---

## Prérequis

- **Ollama** doit tourner localement (`ollama serve`) et le modèle choisi doit être
  téléchargé (`ollama pull mistral`). L'application vérifie la disponibilité et
  affiche un avertissement sinon.
- Les données restent locales : aucune clé API ni service cloud requis.
