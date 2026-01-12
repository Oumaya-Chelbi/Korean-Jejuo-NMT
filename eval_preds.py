import re
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

# --- Fonction pour nettoyer le texte ---
def clean_korean(text):
    """
    Garde uniquement les caractères coréens (Hangul) et les espaces.
    """
    return ' '.join(re.findall(r'[가-힣]+', text))

# --- Charger les fichiers ---
def load_file(file_path):
    """
    Charge un fichier txt et retourne une liste de phrases nettoyées.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    cleaned_lines = [clean_korean(line.strip()) for line in lines]
    return cleaned_lines

# --- Chemins vers tes fichiers ---
pred_file = 'predictions_test.txt'
gt_file = 'txt/je_test.txt'

# --- Charger et nettoyer les données ---
preds = load_file(pred_file)
refs = load_file(gt_file)

# --- Ligne spécifique à calculer (None = tout le corpus) ---
line_index = None  # Exemple : 42 pour la ligne 43, ou None pour tout le corpus

if line_index is None:
    # --- Calcul BLEU pour tout le corpus ---
    refs_tokenized = [[ref.split()] for ref in refs]
    preds_tokenized = [pred.split() for pred in preds]
    bleu_score = corpus_bleu(refs_tokenized, preds_tokenized)
    print(f"BLEU score (corpus entier): {bleu_score}")
else:
    # --- Vérifier que l'index est valide ---
    if line_index < 0 or line_index >= len(preds):
        raise ValueError(f"line_index doit être entre 0 et {len(preds)-1}")
    # BLEU pour une seule phrase
    reference = [refs[line_index].split()]  # list of list pour sentence_bleu
    hypothesis = preds[line_index].split()
    bleu_score_line = sentence_bleu(reference, hypothesis)
    print(f"BLEU score (ligne {line_index}): {bleu_score_line}")
