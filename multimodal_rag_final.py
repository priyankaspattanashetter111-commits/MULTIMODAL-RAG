
import os
import warnings

import chromadb
import ollama
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD THE FLOWER DATASET
# ==========================================

print("Loading flower dataset...")

ds = load_dataset("huggan/flowers-102-categories")

dataset_folder = "./dataset/flowers-102-categories"
os.makedirs(dataset_folder, exist_ok=True)

# ==========================================
# 2. SAVE FLOWER IMAGES
# ==========================================

def save_images(dataset, folder, num_images=500):
    total = min(num_images, len(dataset["train"]))

    for i in range(total):
        image_path = os.path.join(folder, f"flower_{i+1}.png")

        if not os.path.exists(image_path):
            image = dataset["train"][i]["image"]
            image.save(image_path)

        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"Prepared {i+1}/{total} images")

    print("Flower images are ready.")


save_images(ds, dataset_folder, num_images=500)

# ==========================================
# 3. SET UP CHROMADB AND OPENCLIP
# ==========================================

print("\nInitializing ChromaDB and OpenCLIP...")

chroma_client = chromadb.PersistentClient(path="./data/flower.db")

image_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()

flower_collection = chroma_client.get_or_create_collection(
    name="flowers_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

# Collect image paths
image_paths = []

for filename in sorted(os.listdir(dataset_folder)):
    if filename.lower().endswith(".png"):
        image_paths.append(
            os.path.abspath(os.path.join(dataset_folder, filename))
        )

# Index images only when the collection is empty
if flower_collection.count() == 0:
    print("Adding flower images to ChromaDB...")

    batch_size = 50

    for start in range(0, len(image_paths), batch_size):
        batch = image_paths[start:start + batch_size]

        flower_collection.add(
            ids=[os.path.basename(path) for path in batch],
            uris=batch,
        )

        print(
            f"Indexed {min(start + len(batch), len(image_paths))}"
            f"/{len(image_paths)} images"
        )

print("Images in ChromaDB:", flower_collection.count())

# ==========================================
# 4. IMAGE DISPLAY FUNCTION
# ==========================================

def show_image_from_uri(uri):
    with Image.open(uri) as image:
        plt.figure(figsize=(5, 4))
        plt.imshow(image.convert("RGB"))
        plt.axis("off")
        plt.show()

# ==========================================
# 5. RETRIEVE RELEVANT IMAGES
# ==========================================

def query_db(query, results=2):
    total = flower_collection.count()

    if total == 0:
        return []

    results = flower_collection.query(
        query_texts=[query],
        n_results=min(results, total),
        include=["uris", "distances"],
    )

    return [
        {
            "uri": uri,
            "distance": distance,
            "id": image_id,
        }
        for image_id, uri, distance in zip(
            results["ids"][0],
            results["uris"][0],
            results["distances"][0],
        )
    ]

# ==========================================
# 6. GENERATE RESPONSE USING LOCAL GEMMA 3
# ==========================================

def generate_bouquet_suggestions(query, retrieved_images):
    if not retrieved_images:
        return "No matching flower images were found."

    print("\nAnalyzing retrieved images with local Gemma 3...")

    image_paths = [
        result["uri"] for result in retrieved_images
    ]

    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a helpful florist and bouquet designer.\n\n"
                    f"Customer request: {query}\n\n"
                    "Analyze the attached flower images. Suggest a "
                    "suitable bouquet arrangement based on the visible "
                    "colors and flower appearance. Explain how to "
                    "combine the flowers, suitable occasions, and "
                    "complementary colors. Do not claim a flower species "
                    "is certain if it cannot be identified reliably. "
                    "Clearly state that your suggestions are based on "
                    "the retrieved images."
                ),
                "images": image_paths,
            }
        ],
    )

    return response["message"]["content"]

# ==========================================
# 7. RUN THE MULTIMODAL RAG APPLICATION
# ==========================================

def main():
    print("\n======================================")
    print("   FLOWER ARRANGEMENT ASSISTANT")
    print("   Powered by Local Multimodal RAG")
    print("======================================")

    print("\nEnter a flower or bouquet-related query.")
    query = input("Your query: ").strip()

    if not query:
        print("Please enter a valid query.")
        return

    # Retrieval
    retrieved_images = query_db(query, results=2)

    if not retrieved_images:
        print("No images found in the database.")
        return

    print("\nRetrieved flower images:")

    for index, item in enumerate(retrieved_images, start=1):
        print(f"{index}. {item['uri']}")
        print(f"   Distance: {item['distance']}")

    # Generation
    try:
        answer = generate_bouquet_suggestions(
            query, retrieved_images
        )

        print("\n========== BOUQUET SUGGESTIONS ==========\n")
        print(answer)

    except Exception as error:
        print("\nCould not generate suggestions.")
        print("Error:", error)
        print(
            "Check that Ollama is running and gemma3:4b "
            "is available using: ollama list"
        )
        return

    # Display retrieved images
    print("\n========== RETRIEVED IMAGES ==========")

    for index, item in enumerate(retrieved_images, start=1):
        print(f"Displaying image {index}")
        show_image_from_uri(item["uri"])


if __name__ == "__main__":
    main()