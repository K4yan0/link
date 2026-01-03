from huggingface_hub import hf_hub_download
import os

# On change pour le repo de Bartowski (très fiable)
REPO_ID = "bartowski/Qwen2.5-7B-Instruct-GGUF"
# Attention aux majuscules, c'est précis !
FILENAME = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
MODEL_DIR = "models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print(f"--- Début du téléchargement de {FILENAME} ---")
print(f"Source : {REPO_ID}")
print("Cela peut prendre quelques minutes (4.7 Go)...")

try:
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False
    )
    print(f"\n✅ SUCCÈS ! Modèle téléchargé ici : {model_path}")
except Exception as e:
    print(f"\n❌ ERREUR : {e}")