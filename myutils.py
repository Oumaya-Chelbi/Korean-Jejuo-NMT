from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
from keras_hub.layers import TokenAndPositionEmbedding, TransformerEncoder, TransformerDecoder
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, Dropout, Dense
from tensorflow.keras.models import Model

def load_data_from_txt(file_path, num_samples=None):
    """
    Charger un fichier texte et ajouter <BOS> et <EOS> à chaque ligne.
    
    :param file_path: chemin du fichier texte
    :param num_samples: nombre d'exemples à charger (par défaut None pour charger toutes les lignes)
    :return: liste des lignes avec <BOS> et <EOS> ajoutés
    """
    print(f"Chargement du fichier : {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()  # Lire toutes les lignes du fichier
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{file_path}' n'a pas été trouvé.")
        return []
    except Exception as e:
        print(f"Erreur lors de l'ouverture du fichier : {e}")
        return []

    print(f"Nombre de lignes dans {file_path}: {len(lines)}")
    
    # Retirer les éventuels sauts de ligne à la fin de chaque ligne
    lines = [line.strip() for line in lines if line.strip() != '']
    
    # Ajouter <BOS> au début et <EOS> à la fin de chaque ligne
    lines_with_bos_eos = ['<BOS> ' + line + ' <EOS>' for line in lines]
    
    # Vérifier si num_samples est supérieur au nombre de lignes disponibles
    if num_samples is not None and num_samples > len(lines_with_bos_eos):
        print(f"Avertissement : Demande plus d'exemples ({num_samples}) que disponibles ({len(lines_with_bos_eos)}). Retourne tout.")
    
    # Ne garder que les num_samples premières lignes (par défaut toutes les lignes)
    return lines_with_bos_eos if num_samples is None else lines_with_bos_eos[:num_samples]

def tokenize_with_tfidf(texts, max_features=None, stop_words=None):
    """
    Tokenisation des textes en utilisant la méthode TF-IDF
    :param texts: Liste des textes à tokeniser
    :param max_features: Le nombre maximum de caractéristiques (mots) à prendre en compte (par défaut None pour ne pas limiter)
    :param stop_words: Liste de mots vides à ignorer (par défaut None pour ne pas utiliser de stopwords)
    :return: La matrice TF-IDF pour les textes et les mots associés
    """
    print("Tokenisation avec TF-IDF...")
    
    # Créer l'objet TfidfVectorizer avec ou sans limitation de max_features
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    
    # Appliquer le fit_transform pour calculer le TF-IDF et obtenir la matrice
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Obtenir les mots correspondants aux colonnes de la matrice TF-IDF
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"Nombre de caractéristiques (mots) : {len(feature_names)}")
    
    return tfidf_matrix, feature_names

def create_sequences(texts, max_len=None, tokenizer=None):
    """
    Crée des séquences à partir des textes, en les tokenisant, en créant les entrées et sorties,
    et en effectuant un padding si nécessaire.
    
    :param texts: Liste des textes à transformer (coréen ou jejueo)
    :param max_len: Longueur maximale des séquences. Si None, utilise la longueur de la séquence la plus longue.
    :param tokenizer: Si fourni, on utilise ce tokenizer pour les textes, sinon un nouveau tokenizer est créé.
    :return: Les séquences d'entrée, les séquences de sortie, les séquences après padding et la longueur maximale
    """
    
    # Si aucun tokenizer n'est fourni, on crée un nouveau tokenizer
    if tokenizer is None:
        tokenizer = Tokenizer(char_level=False)
        tokenizer.fit_on_texts(texts)
    
    # Conversion des textes en séquences d'indices
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Si max_len est None, on calcule la longueur maximale des séquences
    if max_len is None:
        max_len = max([len(seq) for seq in sequences])
    
    # Création des entrées et sorties (décalage des séquences)
    input_seq = [seq[:-1] for seq in sequences]  # Décalage pour l'entrée
    output_seq = [seq[1:] for seq in sequences]  # Décalage pour la sortie
    
    # Padding des séquences pour qu'elles aient toutes la même longueur (et coupe si nécessaire)
    sequences_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    input_seq_padded = pad_sequences(input_seq, maxlen=max_len, padding='post', truncating='post')
    output_seq_padded = pad_sequences(output_seq, maxlen=max_len, padding='post', truncating='post')
    
    # Affichage de la longueur maximale et de la taille des séquences après padding
    print(f"Longueur maximale des séquences : {max_len}")
    print(f"Taille des séquences après padding (Entrées) : {input_seq_padded.shape}")
    print(f"Taille des séquences après padding (Sorties) : {output_seq_padded.shape}")
    
    # Retourner les entrées et sorties, ainsi que les séquences après padding et max_len
    return input_seq_padded, output_seq_padded, sequences_padded, max_len, tokenizer

def greedy_decode_lstm(model, encoder_input, jeju_tokenizer, max_len, bos_id=2, eos_id=3):
    """
    Décodage auto-régressif greedy pour un modèle LSTM seq2seq.

    :param model: modèle LSTM entraîné
    :param encoder_input: (1, max_len_kor) séquence coréenne tokenisée
    :param jeju_tokenizer: tokenizer jejueo
    :param max_len: longueur maximale de la séquence de sortie
    :param bos_id: ID du token BOS (par défaut 2)
    :param eos_id: ID du token EOS (par défaut 3)
    :return: phrase jejueo décodée (string)
    """

    # Initialisation du décodeur avec <BOS>
    decoder_input = np.array([[bos_id]])

    decoded_token_ids = []

    for _ in range(max_len):
        # Prédiction du modèle
        predictions = model.predict(
            [encoder_input, decoder_input],
            verbose=0
        )

        # On récupère le dernier timestep
        next_token_id = int(tf.argmax(predictions[0, -1]).numpy())
        #print(f"Predicted token ID: {next_token_id}")

        # Arrêt si <EOS>
        if next_token_id == eos_id:
            break

        decoded_token_ids.append(next_token_id)

        # Ajout du token prédit à l'entrée du décodeur
        decoder_input = np.concatenate(
            [decoder_input, [[next_token_id]]],
            axis=1
        )

    # Conversion IDs → mots
    decoded_words = [
        jeju_tokenizer.index_word.get(token_id, "")
        for token_id in decoded_token_ids
    ]

    return " ".join(decoded_words)

def masked_sparse_categorical_crossentropy(y_true, y_pred, pad_id=0):
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=False
    )
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

class MyTransformer:
    def __init__(self, kor_tokenizer, jeju_tokenizer, embedding_dim=256, num_heads=8, ff_dim=512, num_layers=4, max_len_kor=29, max_len_jeju=29):
        """
        Initialisation du modèle Transformer
        :param kor_tokenizer: Le tokenizer pour les textes en coréen
        :param jeju_tokenizer: Le tokenizer pour les textes en jejueo
        :param embedding_dim: Dimension des embeddings
        :param num_heads: Nombre de têtes dans le mécanisme d'attention multi-tête
        :param ff_dim: Dimension de la couche feed-forward
        :param num_layers: Nombre de couches dans l'encodeur et le décodeur
        :param max_len_kor: Longueur maximale des séquences coréennes
        :param max_len_jeju: Longueur maximale des séquences jejueo
        """
        self.kor_tokenizer = kor_tokenizer
        self.jeju_tokenizer = jeju_tokenizer
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.max_len_kor = max_len_kor
        self.max_len_jeju = max_len_jeju
        
        # Récupérer la taille du vocabulaire dynamiquement à partir des tokenizers
        self.vocab_size_kor = len(self.kor_tokenizer.word_index) + 1  # +1 pour tenir compte du token OOV
        self.vocab_size_jeju = len(self.jeju_tokenizer.word_index) + 1  # +1 pour tenir compte du token OOV
        
        # Créer le modèle Transformer
        self.model = self.create_transformer_model()
    
    def transformer_encoder(self, inputs):
        """ Encoder Transformer """
        x = Embedding(self.vocab_size_kor, self.embedding_dim)(inputs)
        for _ in range(self.num_layers):
            # Mécanisme d'attention multi-tête
            attn = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embedding_dim)(x, x)
            attn = Dropout(0.1)(attn)
            x = LayerNormalization()(x + attn)

            # Couche Feed Forward
            ff = Dense(self.ff_dim, activation='relu')(x)
            ff = Dense(self.embedding_dim)(ff)
            ff = Dropout(0.1)(ff)
            x = LayerNormalization()(x + ff)
        return x

    def transformer_decoder(self, inputs, encoder_output):
        """ Decoder Transformer """
        x = Embedding(self.vocab_size_jeju, self.embedding_dim)(inputs)
        for _ in range(self.num_layers):
            # Attention sur l'encodeur
            attn = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embedding_dim)(x, encoder_output)
            attn = Dropout(0.1)(attn)
            x = LayerNormalization()(x + attn)

            # Couche Feed Forward
            ff = Dense(self.ff_dim, activation='relu')(x)
            ff = Dense(self.embedding_dim)(ff)
            ff = Dropout(0.1)(ff)
            x = LayerNormalization()(x + ff)
        return x

    def create_transformer_model(self):
        """ Crée le modèle Transformer complet """
        encoder_input = Input(shape=(self.max_len_kor,))
        decoder_input = Input(shape=(self.max_len_jeju,))

        encoder_output = self.transformer_encoder(encoder_input)
        decoder_output = self.transformer_decoder(decoder_input, encoder_output)

        output = Dense(self.vocab_size_jeju, activation='softmax')(decoder_output)

        model = Model(inputs=[encoder_input, decoder_input], outputs=output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model
    
    def get_model(self):
        """ Retourne le modèle Transformer """
        return self.model

class HubTransformer:
    def __init__(self, kor_tokenizer, jeju_tokenizer, intermediate_dim=256, num_heads=8, num_layers=4,
                 max_len_kor=29, max_len_jeju=29, dropout=0.1):
        """
        Initialisation du modèle Transformer
        :param kor_tokenizer: Le tokenizer pour les textes en coréen
        :param jeju_tokenizer: Le tokenizer pour les textes en jejueo
        :param intermediate_dim: Dimension des couches intermédiaires (couches feed-forward)
        :param num_heads: Nombre de têtes dans le mécanisme d'attention multi-tête
        :param num_layers: Nombre de couches dans l'encodeur et le décodeur
        :param max_len_kor: Longueur maximale des séquences coréennes
        :param max_len_jeju: Longueur maximale des séquences jejueo
        """
        self.kor_tokenizer = kor_tokenizer
        self.jeju_tokenizer = jeju_tokenizer
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len_kor = max_len_kor
        self.max_len_jeju = max_len_jeju
        self.dropout = dropout
        
        # Récupérer la taille du vocabulaire dynamiquement à partir des tokenizers
        self.vocab_size_kor = len(self.kor_tokenizer.word_index) + 1  # +1 pour le token OOV
        self.vocab_size_jeju = len(self.jeju_tokenizer.word_index) + 1  # +1 pour le token OOV
        
        # Créer le modèle Transformer
        self.model = self.create_transformer_model()
    
    def transformer_encoder(self, inputs):
        """ Crée un encodeur Transformer utilisant keras_hub """
        
        # Embedding de position
        position_embedding = TokenAndPositionEmbedding(
            vocabulary_size=self.vocab_size_jeju,  # vocabulaire du jeju
            sequence_length=self.max_len_jeju,     # longueur maximale de la séquence jeju
            embedding_dim=self.intermediate_dim,   # dimension de l'embedding
        )(inputs)
    
        # Encoder Transformer (pas de calcul explicite de head_dim)
        encoder_output = TransformerEncoder(
            intermediate_dim=self.intermediate_dim, 
            num_heads=self.num_heads, 
            dropout=self.dropout
        )(position_embedding)
        print("encoder_output shape:", encoder_output.shape)
        return encoder_output
    
    
    def transformer_decoder(self, decoder_inputs, encoder_inputs):
        """ Crée un décodeur Transformer utilisant keras_hub """
        
        # Embedding de position pour le décodeur
        position_embedding = TokenAndPositionEmbedding(
            vocabulary_size=self.vocab_size_jeju,  # vocabulaire du jeju
            sequence_length=self.max_len_jeju,     # longueur maximale de la séquence jejueo
            embedding_dim=self.intermediate_dim,   # dimension de l'embedding
        )(decoder_inputs)
        print("decoder_emb shape:", position_embedding.shape)
        
        # Decoder Transformer sans manipulations manuelles de dimensions
        decoder_output = TransformerDecoder(
            intermediate_dim=self.intermediate_dim, 
            num_heads=self.num_heads, 
            dropout=self.dropout
        )([position_embedding, encoder_inputs])  # Attention croisée avec l'encodeur
    
        return decoder_output


    def create_transformer_model(self):
        """ Crée le modèle Transformer complet """
        # Entrées pour l'encodeur et le décodeur
        encoder_input = Input(shape=(self.max_len_kor,), dtype="int32", name="decoder_input")
        decoder_input = Input(shape=(self.max_len_jeju,), dtype="int32", name="decoder_input")
        print("encoder_input shape:", encoder_input.shape)
        print("decoder_input shape:", decoder_input.shape)

        # Passage dans l'encodeur
        encoder_output = self.transformer_encoder(encoder_input)

        # Passage dans le décodeur avec l'input de l'encodeur
        decoder_output = self.transformer_decoder(decoder_input, encoder_output)

        # Couche finale pour générer la prédiction
        output = Dense(self.vocab_size_jeju, activation='softmax')(decoder_output)

        # Créer le modèle Keras
        model = Model(inputs=[encoder_input, decoder_input], outputs=output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def get_model(self):
        """ Retourne le modèle Transformer """
        return self.model

class LSTMModel:
    def __init__(self, kor_tokenizer, jeju_tokenizer, embedding_dim=256, latent_dim=256, max_len_kor=29, max_len_jeju=29):
        """
        Initialisation du modèle LSTM
        :param kor_tokenizer: Le tokenizer pour les textes en coréen
        :param jeju_tokenizer: Le tokenizer pour les textes en jejueo
        :param embedding_dim: Dimension des embeddings
        :param latent_dim: Dimension des couches LSTM (latentes)
        :param max_len_kor: Longueur maximale des séquences coréennes
        :param max_len_jeju: Longueur maximale des séquences jejueo
        """
        self.kor_tokenizer = kor_tokenizer
        self.jeju_tokenizer = jeju_tokenizer
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.max_len_kor = max_len_kor
        self.max_len_jeju = max_len_jeju
        
        # Créer le modèle LSTM
        self.model = self.create_lstm_model()

    def create_lstm_model(self):
        """
        Crée le modèle LSTM complet
        """
        # Entrées pour l'encodeur et le décodeur
        encoder_input = layers.Input(shape=(self.max_len_kor,))
        decoder_input = layers.Input(shape=(self.max_len_jeju,))

        # Embedding pour l'encodeur (Coréen)
        encoder_embedding = layers.Embedding(input_dim=len(self.kor_tokenizer.word_index) + 1,
                                             output_dim=self.embedding_dim)(encoder_input)
        encoder_lstm = layers.LSTM(self.latent_dim, return_state=True)
        encoder_output, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        # Embedding pour le décodeur (Jejueo)
        decoder_embedding = layers.Embedding(input_dim=len(self.jeju_tokenizer.word_index) + 1,
                                             output_dim=self.embedding_dim)(decoder_input)
        decoder_lstm = layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_lstm_output, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

        # Couche dense pour la sortie du décodeur
        decoder_dense = layers.Dense(len(self.jeju_tokenizer.word_index) + 1, activation='softmax')
        decoder_output = decoder_dense(decoder_lstm_output)

        # Créer et compiler le modèle
        model = models.Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)
        #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='adam', loss=masked_sparse_categorical_crossentropy, metrics=['accuracy'])
        return model

    def get_model(self):
        """
        Retourne le modèle LSTM
        """
        return self.model

class LSTM_MultiHeadModel:
    def __init__(self, kor_tokenizer, jeju_tokenizer, embedding_dim=256, latent_dim=256, num_heads=4,
                 max_len_kor=29, max_len_jeju=29):
        """
        Initialisation du modèle LSTM + Multi-Head Attention
        :param kor_tokenizer: tokenizer coréen
        :param jeju_tokenizer: tokenizer jejueo
        :param embedding_dim: dimension des embeddings
        :param latent_dim: dimension LSTM
        :param num_heads: nombre de têtes d'attention
        :param max_len_kor: longueur max source
        :param max_len_jeju: longueur max cible
        """
        self.kor_tokenizer = kor_tokenizer
        self.jeju_tokenizer = jeju_tokenizer
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.max_len_kor = max_len_kor
        self.max_len_jeju = max_len_jeju

        self.model = self.create_lstm_model()

    def create_lstm_model(self):
        """
        Crée le modèle LSTM seq2seq avec Multi-Head Attention
        """

        # ===== Inputs =====
        encoder_input = layers.Input(shape=(self.max_len_kor,))
        decoder_input = layers.Input(shape=(self.max_len_jeju,))

        # ===== Encoder =====
        encoder_embedding = layers.Embedding(
            input_dim=len(self.kor_tokenizer.word_index) + 1,
            output_dim=self.embedding_dim
        )(encoder_input)

        encoder_lstm = layers.LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True
        )

        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        # ===== Decoder =====
        decoder_embedding = layers.Embedding(
            input_dim=len(self.jeju_tokenizer.word_index) + 1,
            output_dim=self.embedding_dim
        )(decoder_input)

        decoder_lstm = layers.LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True
        )

        decoder_outputs, _, _ = decoder_lstm(
            decoder_embedding,
            initial_state=encoder_states
        )

        # ===== Multi-Head Cross Attention =====
        attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.latent_dim
        )

        attention_output = attention(
            query=decoder_outputs,
            value=encoder_outputs,
            key=encoder_outputs
        )

        # ===== Fusion =====
        concat = layers.Concatenate(axis=-1)(
            [decoder_outputs, attention_output]
        )

        # ===== Output =====
        decoder_dense = layers.Dense(
            len(self.jeju_tokenizer.word_index) + 1,
            activation="softmax"
        )

        decoder_output = decoder_dense(concat)

        # ===== Model =====
        model = models.Model(
            inputs=[encoder_input, decoder_input],
            outputs=decoder_output
        )

        #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='adam', loss=masked_sparse_categorical_crossentropy, metrics=['accuracy'])
        return model

    def get_model(self):
        """
        Retourne le modèle LSTM
        """
        return self.model

def positional_encoding(seq_len, d_model):
    angle_rates = 1 / np.power(
        10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model)
    )
    positions = np.arange(seq_len)[:, np.newaxis]
    angle_rads = positions * angle_rates[np.newaxis, :]

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.cast(angle_rads[np.newaxis, ...], tf.float32)

def create_padding_mask(seq):
    return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(d_model)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask=None, training=False):
        attn = self.mha(x, x, attention_mask=mask)
        x = self.norm1(x + self.dropout1(attn, training=training))
        ffn = self.ffn(x)
        return self.norm2(x + self.dropout2(ffn, training=training))

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        self.cross_mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(d_model)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None, training=False):
        attn1 = self.self_mha(
            x, x, attention_mask=look_ahead_mask
        )
        x = self.norm1(x + self.dropout(attn1, training=training))

        attn2 = self.cross_mha(
            x, enc_output, attention_mask=padding_mask
        )
        x = self.norm2(x + self.dropout(attn2, training=training))

        ffn = self.ffn(x)
        return self.norm3(x + self.dropout(ffn, training=training))

class UltimateTransformer(tf.keras.Model):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        max_len_src,
        max_len_tgt,
        d_model=256,
        num_heads=4,
        ff_dim=512,
        num_layers=4
    ):
        super().__init__()

        self.src_embedding = tf.keras.layers.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = tf.keras.layers.Embedding(tgt_vocab_size, d_model)

        self.pos_enc_src = positional_encoding(max_len_src, d_model)
        self.pos_enc_tgt = positional_encoding(max_len_tgt, d_model)

        self.encoder_layers = [
            EncoderBlock(d_model, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.decoder_layers = [
            DecoderBlock(d_model, num_heads, ff_dim)
            for _ in range(num_layers)
        ]

        self.final_dense = tf.keras.layers.Dense(tgt_vocab_size)
        
        # self.compile(
        #     optimizer=tf.keras.optimizers.Adam(),
        #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #     metrics=["accuracy"]
        #     )
        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=masked_sparse_categorical_crossentropy,
            metrics=["accuracy"]
            )

    def call(self, inputs, training=False):
        src, tgt = inputs

        enc_padding_mask = create_padding_mask(src)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tgt)[1])
        dec_padding_mask = create_padding_mask(src)

        x = self.src_embedding(src) + self.pos_enc_src[:, :tf.shape(src)[1], :]
        for layer in self.encoder_layers:
            x = layer(x, mask=enc_padding_mask, training=training)

        y = self.tgt_embedding(tgt) + self.pos_enc_tgt[:, :tf.shape(tgt)[1], :]
        for layer in self.decoder_layers:
            y = layer(y, enc_output=x, look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask,
                      training=training)

        return self.final_dense(y)
    
    def get_model(self):
        """
        Retourne le modèle Transformer (pour compatibilité avec le reste du pipeline)
        """
        return self

