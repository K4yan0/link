import sys
from llama_cpp import Llama

# --- CONFIGURATION ---
MODEL_PATH = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

print("🧠 Chargement du cerveau (Version Finale Textuelle)...")

try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_gpu_layers=0,
        verbose=False
    )
except Exception as e:
    print(f"❌ Erreur : {e}")
    sys.exit(1)

# --- PROMPT V4 : ROMANISATION & STABILITÉ ---
SYSTEM_PROMPT = """
RÔLE: Tu es un Ami Polyglotte et Professeur. Langues : Français, Anglais, Japonais, Coréen.

RÈGLES D'AFFICHAGE (OBLIGATOIRES) :
1. FORMAT ASIATIQUE : Pour tout texte en Japonais ou Coréen, tu DOIS ajouter la romanisation entre parenthèses.
   - Exemple Japonais : こんにちは (Konnichiwa)
   - Exemple Coréen : 안녕하세요 (Annyeonghaseyo)
   - C'est CRITIQUE pour l'apprentissage de l'utilisateur.

2. STABILITÉ DE LA LANGUE :
   - Si l'user te parle en Français -> Réponds en Français. (N'utilise pas de mots anglais comme "choice").
   - Ne change de langue QUE si l'utilisateur le demande explicitement (ex: "Comment on dit en Japonais ?").
   - Ne donne pas de traduction spontanée si on ne te le demande pas.

3. DYNAMIQUE DE CONVERSATION :
   - Si l'user joue un rôle (ex: serveur au resto) -> JOUE LE JEU à fond. Ne corrige que les grosses fautes qui empêchent la compréhension.
   - Si l'user fait une petite faute -> Reformule sa phrase correctement dans ta réponse de manière naturelle (Correction implicite).
"""

def chat_loop():
    history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("\n✅ SENSEI PRÊT (Romanisation activée) !")
    print("------------------------------------------------")

    while True:
        try:
            user_input = input("\nToi : ")
            if user_input.lower() in ["exit", "quit"]:
                break

            history.append({"role": "user", "content": user_input})

            print("Sensei : (écrit...)", end="\r")

            output = llm.create_chat_completion(
                messages=history,
                temperature=0.65,
                max_tokens=300,
                repeat_penalty=1.1
            )

            response_text = output['choices'][0]['message']['content']
            print(f"Sensei : {response_text}" + " " * 20)

            history.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    chat_loop()