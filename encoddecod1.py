# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 14:22:27 2026

@author: Alexis
"""
import pickle
import tensorflow as tf
import numpy as np
from myutils import (load_data_from_txt, tokenize_with_tfidf, create_sequences, MyTransformer,
                     HubTransformer, LSTMModel, greedy_decode_lstm, LSTM_MultiHeadModel, UltimateTransformer)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

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
'''
# =========================
# Appliquer la tokenisation avec TF-IDF
# =========================

# Appliquer TF-IDF pour le coréen
kor_train_tfidf, kor_feature_names = tokenize_with_tfidf(kor_train_texts)
kor_dev_tfidf, _ = tokenize_with_tfidf(kor_dev_texts)
kor_test_tfidf, _ = tokenize_with_tfidf(kor_test_texts)

# Appliquer TF-IDF pour le jejueo
jeju_train_tfidf, jeju_feature_names = tokenize_with_tfidf(jeju_train_texts)
jeju_dev_tfidf, _ = tokenize_with_tfidf(jeju_dev_texts)
jeju_test_tfidf, _ = tokenize_with_tfidf(jeju_test_texts)


# Affichage des résultats pour comprendre ce qui se passe
print(f"\nMots extraits pour le coréen (Train) : {kor_feature_names}")
print(f"\nMots extraits pour le jejueo (Train) : {jeju_feature_names}")
# Nombre de termes dans le vocabulaire pour chaque dataset
print(f"Nombre de termes pour le coréen (Train) : {len(kor_feature_names)}")
print(f"Nombre de termes pour le jejueo (Train) : {len(jeju_feature_names)}")
'''
# =========================
# Fusionner les ensembles pour entraîner les tokenizers
# =========================

# Fusionner les datasets (train + dev + test) pour chaque langue
kor_all_texts = kor_train_texts + kor_dev_texts + kor_test_texts
jeju_all_texts = jeju_train_texts + jeju_dev_texts + jeju_test_texts

# =========================
# Entraîner les tokenizers
# =========================

# Créer un tokenizer avec un token OOV pour les mots inconnus
kor_tokenizer = Tokenizer(oov_token="<OOV>")
jeju_tokenizer = Tokenizer(oov_token="<OOV>")

# Appliquer la tokenisation sur tous les textes combinés (train, dev, test)
kor_tokenizer.fit_on_texts(kor_all_texts)
jeju_tokenizer.fit_on_texts(jeju_all_texts)

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
#transformer_model = LSTMModel(kor_tokenizer=kor_tokenizer,
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
#transformer_model = UltimateTransformer(src_vocab_size=len(kor_tokenizer.word_index) + 1,
#                                        tgt_vocab_size=len(jeju_tokenizer.word_index) + 1,
#                                        max_len_src=kor_max_len,
#                                        max_len_tgt=jeju_max_len)

# Obtenir le modèle
model = transformer_model.get_model()
early_stopping = EarlyStopping(
    monitor='val_loss',      # Surveiller la perte sur l'ensemble de validation
    patience=3,              # Attendre 3 époques avant de stopper l'entraînement si pas d'amélioration
    restore_best_weights=True  # Restaurer les poids du modèle au meilleur état
)
# Visualiser le résumé du modèle
model.summary()

#Sauvegarde du vocabulaire
with open("kor_tokenizer.pkl", "wb") as f:
    pickle.dump(kor_tokenizer, f)

with open("jeju_tokenizer.pkl", "wb") as f:
    pickle.dump(jeju_tokenizer, f)

# Entraîner le modèle
model.fit(
    [kor_train_input, jeju_train_input], 
    jeju_train_output,
    validation_data=([kor_dev_input, jeju_dev_input], jeju_dev_output),
    epochs=10, 
    batch_size=16,
    callbacks=[early_stopping])

# Sauvegarder les poids du modèle avec la bonne extension
print("Sauvegarde des poids du modèle...")
model.save_weights('transformer_translation_weights.weights.h5')

# Charger le modèle sauvegardé
pred_model = transformer_model
model = pred_model.get_model()
model.load_weights('transformer_translation_weights.weights.h5')

for i in range(10):
    translation = greedy_decode_lstm(
        model=model,
        encoder_input=kor_test_input[i:i+1],
        jeju_tokenizer=jeju_tokenizer,
        max_len=jeju_max_len
    )

    print("KO :", kor_test_texts[i])
    print("JE :", translation)
    print("-" * 50)

