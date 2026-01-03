import os
import sys
import speech_recognition as sr
from faster_whisper import WhisperModel
from llama_cpp import Llama

# --- CONFIGURATION ---
# Vérifie bien que ce fichier existe !
MODEL_LLM_PATH = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
WHISPER_SIZE = "medium"
TEMP_AUDIO_FILE = "temp_audio.wav"

# --- 1. CHARGEMENT DES MODÈLES ---
print("⏳ Initialisation du système...")

print("   1/2 Chargement de l'Oreille (Whisper)...")
try:
    ear_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
except Exception as e:
    print(f"❌ Erreur Whisper : {e}")
    sys.exit(1)

print("   2/2 Chargement du Cerveau (LLM)...")
try:
    # n_ctx=2048 est suffisant et plus rapide que 4096
    brain_model = Llama(
        model_path=MODEL_LLM_PATH,
        n_ctx=2048,
        n_gpu_layers=0,
        verbose=False
    )
except Exception as e:
    print(f"❌ Erreur LLM : {e}")
    sys.exit(1)

# PROMPT SYSTÈME AMÉLIORÉ (V2)
SYSTEM_PROMPT = """
RÔLE: Tu es "Sensei", un professeur de langue oral expert (Français, Anglais, Japonais, Coréen).

TES DIRECTIVES PÉDAGOGIQUES :
1. ANALYSE L'INTENTION :
   - Si l'user demande une traduction -> Traduis directement sans blabla.
   - Si l'user essaie de parler la langue -> Corrige-le.

2. RÈGLES DE POLITESSE (CRITIQUE) :
   - Japonais/Coréen : Force toujours le registre "Poli Standard" (Desu/Masu, Yo).
   - SI l'user est vulgaire ou impoli (ex: "Omae", "Baka") -> NE TRADUIS PAS. Dis-lui gentiment que c'est inapproprié.

3. FORMAT DE RÉPONSE (Oral) :
   - Fais des réponses COURTES (1 ou 2 phrases max).
   - Ne répète pas systématiquement "Je comprends ce que tu veux dire". Varie tes réponses.
   - Si tu corriges, donne la phrase correcte et demande de répéter.
"""

history = [{"role": "system", "content": SYSTEM_PROMPT}]
recognizer = sr.Recognizer()

# --- 2. BOUCLE DE CONVERSATION ---
print("\n" + "="*50)
print("🎙️  SENSEI EST PRÊT ! (Parlez dans le micro)")
print("="*50 + "\n")

while True:
    try:
        # A. Écoute du microphone
        with sr.Microphone() as source:
            if len(history) == 1:
                print("Calibrage du micro (silence svp)...")
                recognizer.adjust_for_ambient_noise(source, duration=1)

            print("\n👂 J'écoute... (Parlez maintenant)")
            audio_data = recognizer.listen(source, timeout=None)

            print("⏳ Traitement audio...")
            with open(TEMP_AUDIO_FILE, "wb") as f:
                f.write(audio_data.get_wav_data())

        # B. Transcription
        segments, info = ear_model.transcribe(TEMP_AUDIO_FILE, beam_size=5)
        user_text = "".join([segment.text for segment in segments]).strip()

        if not user_text:
            print("⚠️ Je n'ai rien entendu.")
            continue

        print(f"📝 Vous avez dit ({info.language}) : \033[96m{user_text}\033[0m")

        if any(word in user_text.lower() for word in ["stop", "quitter", "exit"]):
            print("Au revoir !")
            break

        # C. Réflexion (LLM)
        history.append({"role": "user", "content": user_text})

        print("🤖 Sensei réfléchit...", end="\r")
        output = brain_model.create_chat_completion(
            messages=history,
            temperature=0.6, # Plus bas = plus précis, moins d'hallucinations
            max_tokens=150   # Réponses plus courtes pour aller plus vite
        )
        response = output['choices'][0]['message']['content']

        # D. Réponse
        print(f"🤖 Sensei : \033[92m{response}\033[0m")
        history.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        print("\nArrêt manuel.")
        break
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        # Si c'est PyAudio qui manque, on le saura ici
        if "PyAudio" in str(e):
            print("💡 Conseil : Essaie 'pip install pipwin' puis 'pipwin install pyaudio'")

if os.path.exists(TEMP_AUDIO_FILE):
    os.remove(TEMP_AUDIO_FILE)