# Korean-Jejuo-NMT


## Projet LSTM MHA / Encodage–Décodage

Ce dépôt contient le code développé dans le cadre du projet des cours DL(TALA536B) et MT(TALA526B)  pour entraîner et évaluer un modèle de type encodeur–décodeur (LSTM avec mha) pour faire de la traduction du coréen vers le jejuo. L’objectif est de montrer à la fois les résultats finaux et tout le cheminement expérimental qui a mené à ces résultats.

## Organisation du Dépôt : 

*Scripts*

*encoddecod1.py:*
Script principal d’entraînement du modèle encodeur–décodeur (chargement des données, définition du modèle, boucle d’entraînement, sauvegarde des poids etc ..).

*make_preds.py:*
Script de génération de prédictions à partir d’un modèle déjà entraîné (chargement des poids, inference sur le jeu de test, sauvegarde des prédictions dans un fichier).

*eval_preds.py:*
Script d’évaluation des prédictions (chargement des prédictions produites par make_preds.py et des données de référence, calcul de métrique(BLEU)).

*myutils.py:*
Script utilitaire contenant les fonctions de prétraitement, de chargement de données, de gestion des chemins de fichiers, etc., réutilisées par les autres scripts.

*sortie1.log:*
Fichier de log contenant la sortie complète du terminal. Les entraînements ayant été lancés dans une session tmux, il n’est pas possible de remonter très loin dans l’historique. Ce fichier permet donc de conserver l’intégralité de la sortie (logs d’entraînement, métriques, messages d’erreur, etc.) pour pouvoir la consulter à tout moment.

*Fichiers de sortie des scripts*

### Modèles et poids entraînés

*transformer_translation_weights.weights.h5 :* 
Fichiers contenant les poids du modèle. Ce fichier permet de recharger le modèle déjà entraîné pour refaire des prédictions sans relancer tout l’entraînement.

### Prédictions
*predictions_test.txt:*
Fichier produit par make_preds.py.Ils contiennent les sorties du modèle (phrases générées, séquences décodées, classes prédites, etc.) pour les exemples du jeu de test.

*requirements.txt:*
Liste des dépendances Python nécessaires.

### Data : 

Dossier txt/ : \*_dev.txt = données de dev , \*_train.txt = données de train, \*_test.txt = données de test.
je_\*.txt = données pour le jejuo et ko_\*.txt = données pour le coréen.

### Petite Précision : 

Par choix pédagogique, tous les scripts  ont été laissés “bruts” :

Les scripts contienent parfois des morceaux de code qui n’ont finalement pas été utilisés dans les résultats finaux et que l'on a juste commenté (on a mis un # devant).

On peut voir différentes tentatives, variantes d’architectures, tests de fonctions etc ..
Rien n’a été “nettoyé” a posteriori : l’idée est de rendre visible le processus de travail réel, avec ses essais, erreurs et ajustements successifs.Pour que les proffesseurs puissent voir l’acheminement complet du projet (explorations, prototypes, changements de direction) ; la manière dont on est passé des premières idées au modèle final ; les décisions techniques prises en réponse aux problèmes rencontrés (paramètres, architecture, prétraitement, gestion des erreurs, etc.).
