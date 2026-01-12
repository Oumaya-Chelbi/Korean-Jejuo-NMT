import pickle
from myutils import (load_data_from_txt, create_sequences, MyTransformer,
                     HubTransformer, LSTMModel, greedy_decode_lstm, LSTM_MultiHeadModel, UltimateTransformer)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

# =========================
# Chargement des fichiers TXT
# =========================

# Load Train datas
kor_train_texts = load_data_from_txt('txt/ko_train.txt', num_samples=None)
jeju_train_texts = load_data_from_txt('txt/je_train.txt', num_samples=None)

# Load Dev datas
kor_dev_texts = load_data_from_txt('txt/ko_dev.txt', num_samples=None)
jeju_dev_texts = load_data_from_txt('txt/je_dev.txt', num_samples=None)

# Load Test datas
kor_test_texts = load_data_from_txt('txt/ko_test.txt', num_samples=None)
jeju_test_texts = load_data_from_txt('txt/je_test.txt', num_samples=None)

# Créer un tokenizer avec un token OOV pour les mots inconnus
kor_tokenizer = Tokenizer(oov_token="<OOV>")
jeju_tokenizer = Tokenizer(oov_token="<OOV>")

# =========================
# Fusionner les ensembles pour entraîner les tokenizers
# =========================

# Fusionner les datasets (train + dev + test) pour chaque langue
kor_all_texts = kor_train_texts + kor_dev_texts + kor_test_texts
jeju_all_texts = jeju_train_texts + jeju_dev_texts + jeju_test_texts

# =========================
# Importer les tokenizers
# =========================

with open("kor_tokenizer.pkl", "rb") as f:
    kor_tokenizer = pickle.load(f)

with open("jeju_tokenizer.pkl", "rb") as f:
    jeju_tokenizer = pickle.load(f)

# =========================
# Appliquer la tokenisation et créer les séquences
# =========================

# Appliquer la tokenisation et créer les séquences pour le coréen
kor_train_input, kor_train_output, kor_train_padded, kor_max_len, kor_tokenizer = create_sequences(kor_train_texts, max_len=None, tokenizer=kor_tokenizer)
jeju_train_input, jeju_train_output, jeju_train_padded, jeju_max_len, jeju_tokenizer = create_sequences(jeju_train_texts, max_len=None, tokenizer=jeju_tokenizer)

# Appliquer la tokenisation et créer les séquences pour la validation (en utilisant les tokenizers déjà entraînés)
kor_dev_input, kor_dev_output, kor_dev_padded, _, _ = create_sequences(kor_dev_texts, max_len=kor_max_len, tokenizer=kor_tokenizer)
jeju_dev_input, jeju_dev_output, jeju_dev_padded, _, _ = create_sequences(jeju_dev_texts, max_len=jeju_max_len, tokenizer=jeju_tokenizer)

# Appliquer la tokenisation et créer les séquences pour le test (en utilisant les tokenizers déjà entraînés)
kor_test_input, kor_test_output, kor_test_padded, _, _ = create_sequences(kor_test_texts, max_len=kor_max_len, tokenizer=kor_tokenizer)
jeju_test_input, jeju_test_output, jeju_test_padded, _, _ = create_sequences(jeju_test_texts, max_len=jeju_max_len, tokenizer=jeju_tokenizer)

# Affichage des résultats pour vérifier que tout fonctionne
print(f"Longueur maximale pour le coréen : {kor_max_len}")
print(f"Longueur maximale pour le jejueo : {jeju_max_len}")
print(f"Taille des séquences coréennes après padding : {kor_train_padded.shape}")
print(f"Taille des séquences jejueo après padding : {jeju_train_padded.shape}")

# Affichage du vocabulaire (optionnel, pour voir quelques exemples de mots et leur indice)
print(f"Vocabulaire coréen (exemples) : {list(kor_tokenizer.word_index.items())[:10]}")
print(f"Vocabulaire jejueo (exemples) : {list(jeju_tokenizer.word_index.items())[:10]}")

# Créer un modèle Transformer
# transformer_model = MyTransformer(kor_tokenizer=kor_tokenizer, jeju_tokenizer=jeju_tokenizer, embedding_dim=256, num_heads=8, ff_dim=512, num_layers=4, max_len_kor=kor_max_len, max_len_jeju=jeju_max_len)
# transformer_model = HubTransformer(kor_tokenizer=kor_tokenizer, 
#                                    jeju_tokenizer=jeju_tokenizer, 
#                                    intermediate_dim=256, 
#                                    num_heads=8, 
#                                    num_layers=4, 
#                                    max_len_kor=kor_max_len, 
#                                    max_len_jeju=jeju_max_len)
# transformer_model = LSTMModel(kor_tokenizer=kor_tokenizer,
#                               jeju_tokenizer=jeju_tokenizer,
#                               embedding_dim=256,
#                               latent_dim=256,
#                               max_len_kor=kor_max_len,
#                               max_len_jeju=jeju_max_len)
transformer_model = LSTM_MultiHeadModel(kor_tokenizer=kor_tokenizer,
                              jeju_tokenizer=jeju_tokenizer,
                              embedding_dim=256,
                              latent_dim=256,
                              num_heads=8,
                              max_len_kor=kor_max_len,
                              max_len_jeju=jeju_max_len)
# transformer_model = UltimateTransformer(src_vocab_size=len(kor_tokenizer.word_index) + 1,
#                                         tgt_vocab_size=len(jeju_tokenizer.word_index) + 1,
#                                         max_len_src=kor_max_len,
#                                         max_len_tgt=jeju_max_len)

# Charger le modèle sauvegardé
pred_model = transformer_model
model = pred_model.get_model()

model.load_weights('transformer_translation_weights.weights.h5')

output_file = "predictions_test.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for i in range(len(kor_test_texts)):
        translation = greedy_decode_lstm(
            model=model,
            encoder_input=kor_test_input[i:i+1],
            jeju_tokenizer=jeju_tokenizer,
            max_len=jeju_max_len
        )

        # Écriture : UNE phrase par ligne, rien d'autre
        f.write(translation.strip() + "\n")

        # Affichage console uniquement pour les 10 premières
        if i < 10:
            print("KO   :", kor_test_texts[i])
            print("PRED :", translation)
            print("-" * 50)

print(f"\nFichier de prédictions généré : {output_file}")