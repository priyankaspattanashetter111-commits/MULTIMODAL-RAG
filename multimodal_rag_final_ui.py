
import os
import warnings
import streamlit as st
import chromadb
import ollama

from PIL import Image
from datasets import load_dataset
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Flower Arrangement Assistant", page_icon="🌸")

st.title("🌸 Flower Arrangement Query and Image Retrieval")
st.write("Find flower images and generate bouquet suggestions using a local AI model.")

DATASET_FOLDER = "./dataset/flowers-102-categories"
os.makedirs(DATASET_FOLDER, exist_ok=True)


@st.cache_data
def load_flower_dataset():
    return load_dataset("huggan/flowers-102-categories")


@st.cache_resource
def setup_database():
    client = chromadb.PersistentClient(path="./data/flower.db")

    collection = client.get_or_create_collection(
        name="flowers_collection",
        embedding_function=OpenCLIPEmbeddingFunction(),
        data_loader=ImageLoader(),
    )

    # Prepare images if the collection is empty
    if collection.count() == 0:
        dataset = load_flower_dataset()
        saved_paths = []

        for i, item in enumerate(dataset["train"]):
            if i >= 500:
                break

            image = item["image"].convert("RGB")
            path = os.path.abspath(
                os.path.join(DATASET_FOLDER, f"flower_{i}.png")
            )
            image.save(path)
            saved_paths.append(path)

        for start in range(0, len(saved_paths), 50):
            batch = saved_paths[start:start + 50]

            collection.add(
                ids=[os.path.basename(path) for path in batch],
                uris=batch,
            )

    return collection


def query_db(collection, query, number=2):
    total = collection.count()

    if total == 0:
        return []

    result = collection.query(
        query_texts=[query],
        n_results=min(number, total),
        include=["uris", "distances"],
    )

    matches = []

    for i, uri in enumerate(result["uris"][0]):
        matches.append({
            "uri": uri,
            "distance": result["distances"][0][i],
        })

    return matches


def generate_suggestions(query, image_paths):
    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable florist. Use the supplied flower "
                    "images and the user's request to suggest attractive bouquet "
                    "arrangements. Explain flower combinations, colors, and "
                    "occasions. Do not claim an image contains a flower unless "
                    "you can identify it from the image."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"My bouquet request is: {query}. "
                    "Please examine the attached images and suggest suitable "
                    "arrangements based on them."
                ),
                "images": image_paths,
            },
        ],
    )

    return response["message"]["content"]


collection = setup_database()
st.success(f"Flower image database ready: {collection.count()} images indexed.")

query = st.text_input(
    "Describe the flowers or bouquet you want",
    placeholder="e.g., A romantic pink and white anniversary bouquet",
)

if st.button("Find Flowers and Suggest a Bouquet", type="primary"):
    if not query.strip():
        st.warning("Please enter a flower or bouquet query.")
    else:
        with st.spinner("Searching flower images..."):
            matches = query_db(collection, query, number=2)

        if not matches:
            st.warning("No images were found in the database.")
        else:
            st.subheader("Retrieved Flower Images")

            image_paths = []

            columns = st.columns(len(matches))

            for i, match in enumerate(matches):
                path = match["uri"]
                image_paths.append(path)

                with columns[i]:
                    st.image(path, use_container_width=True)
                    st.caption(
                        f"Distance: {match['distance']:.4f}"
                    )

            with st.spinner("Gemma 3 is creating bouquet suggestions..."):
                try:
                    answer = generate_suggestions(query, image_paths)

                    st.subheader("💐 Bouquet Suggestions")
                    st.markdown(answer)

                except Exception as error:
                    st.error(
                        "Could not contact the local Ollama model. "
                        "Check that Ollama is running and gemma3:4b is installed."
                    )
                    st.caption(str(error))