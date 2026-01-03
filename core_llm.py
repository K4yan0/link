import sys
from llama_cpp import Llama

# Chemin EXACT vers le modèle (Attention aux majuscules !)
MODEL_PATH = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

print("🧠 Chargement du cerveau en cours... (Cela peut prendre 10-20 secondes)")

try:
    # Initialisation du modèle
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,       # Mémoire de la conversation
        n_gpu_layers=0,   # 0 pour CPU. Si tu as un GPU Nvidia, mets -1 pour aller plus vite.
        verbose=False     # Pour cacher le blabla technique
    )
except Exception as e:
    print(f"❌ Erreur au chargement du modèle : {e}")
    print("Vérifie que le fichier est bien dans le dossier 'models' !")
    sys.exit(1)

# LE COEUR DU PROJET : La consigne pédagogique
SYSTEM_PROMPT = """
RÔLE: Tu es un Tuteur de Langues expert et patient. Tes langues : Français, Anglais, Japonais, Coréen.

RÈGLES DE COMPORTEMENT:
1. NIVEAU DE POLITESSE : 
   - Japonais : Utilise la forme polie (Desu/Masu) UNIQUEMENT. Pas de langage familier, pas de Keigo complexe.
   - Coréen : Utilise la forme polie (Haeyo-che / terminaison en -yo).
2. CORRECTION (Méthode Sandwich) :
   - Si l'user fait une faute, ne dis pas juste "C'est faux".
   - Dis : "Je comprends ce que tu veux dire" -> "Voici la petite erreur" -> "Répète après moi : [Phrase Corrigée]".
3. CONVERSATION :
   - Pose toujours une question à la fin pour relancer la discussion.
   - Si l'user te parle en Français, réponds en Français (et enseigne la langue cible s'il y en a une, sinon converse).
"""

def chat_loop():
    # Historique de la conversation
    history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("\n✅ PROFESSEUR PRÊT ! (Tape 'exit' pour quitter)")
    print("------------------------------------------------")

    while True:
        try:
            user_input = input("\nToi : ")
            if user_input.lower() in ["exit", "quit"]:
                print("À bientôt !")
                break

            # Ajout du message user
            history.append({"role": "user", "content": user_input})

            # Génération de la réponse
            print("Prof : (réfléchit...)", end="\r")

            output = llm.create_chat_completion(
                messages=history,
                temperature=0.7, # Créativité
                max_tokens=300   # Longueur max de réponse
            )

            response_text = output['choices'][0]['message']['content']

            # Affichage propre (on écrase le "réfléchit...")
            print(f"Prof : {response_text}" + " " * 20)

            # Sauvegarde dans la mémoire
            history.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            print("\nArrêt forcé.")
            break

if __name__ == "__main__":
    chat_loop()