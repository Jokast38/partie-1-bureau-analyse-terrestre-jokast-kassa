# Bureau d'Analyse Terrestre - Partie 1

Analyse des relevés d'observation collectés par la sonde Klaxo-3, pour trier
les canulars des signalements réels.

## Lancer l'analyse

```bash
pip install pandas numpy scikit-learn
python analyse.py
```

Le script télécharge `releves_klaxo3.csv` s'il est absent, puis rejoue tout le
traitement d'une traite (nettoyage, features, modèles). Le fichier de données
n'est pas versionné dans ce dépôt (15 Mo, téléchargeable).

Voir [RAPPORT.md](RAPPORT.md) pour le détail phase par phase et les chiffres
demandés par le Conseil.
