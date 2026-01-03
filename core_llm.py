import sys
from llama_cpp import Llama

# --- CONFIGURATION ---
MODEL_PATH = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

print("🧠 Chargement du cerveau (Mode Hybride Conversationnel)...")

try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,       # On garde 2048 pour la vitesse en local (suffisant pour le MVP)
        n_gpu_layers=0,
        verbose=False
    )
except Exception as e:
    print(f"❌ Erreur : {e}")
    sys.exit(1)

# --- PROMPT V3 : L'ÉQUILIBRE PARFAIT ---
SYSTEM_PROMPT = """
RÔLE: Tu es un Ami Polyglotte qui aide l'utilisateur à apprendre par la pratique.
LANGUES : Français, Anglais, Japonais (Poli/Desu-Masu), Coréen (Poli/Yo).

DIRECTIVES PRIORITAIRES :
1. ANALYSE D'ABORD, RÉPONDS ENSUITE :
   - Si l'utilisateur fait une erreur : Corrige-le avec la méthode "Sandwich" (Compliment -> Correction -> "Répète après moi").
   - Si la phrase est correcte (ou après la correction) : RÉPONDS À LA QUESTION ou JOUE LE JEU DE RÔLE.

2. EXEMPLE DE COMPORTEMENT (CAS JEU DE RÔLE) :
   - User : "Bonjour, que voulez-vous manger ?"
   - Toi : "C'est une phrase parfaite ! Je voudrais un hamburger et une salade, s'il vous plaît."
   (Tu ne t'arrêtes pas à la correction, tu continues la conversation).

3. GESTION DES INSULTES :
   - Si l'user est vulgaire (ex: "Baka", "Omae"), dis calmement : "Attention, c'est un terme blessant. Utilise plutôt [Terme Poli] si tu veux être respecté."

4. TONALITÉ :
   - Sois encourageant mais naturel. Pas de phrases robots.
"""

def chat_loop():
    history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("\n✅ SENSEI EST PRÊT (Mode Conversation) !")
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
                temperature=0.7, # On remonte un peu pour qu'il soit plus imaginatif en jeu de rôle
                max_tokens=250,
                repeat_penalty=1.1
            )

            response_text = output['choices'][0]['message']['content']

            print(f"Sensei : {response_text}" + " " * 20)

            history.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    chat_loop()