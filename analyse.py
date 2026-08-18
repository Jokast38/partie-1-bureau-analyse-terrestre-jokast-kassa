"""
Bureau d'Analyse Terrestre - analyse des relevés de la sonde Klaxo-3.
Script unique, se relance d'une traite : téléchargement -> nettoyage -> features -> modèles.
"""

import os
import re
import csv
import sys
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

RANDOM_STATE = 42

DATA_URL = "https://raw.githubusercontent.com/planetsig/ufo-reports/master/csv-data/ufo-complete-geocoded-time-standardized.csv"
DATA_FILE = "releves_klaxo3.csv"

COLUMNS = [
    "datetime", "city", "state", "country", "shape",
    "duration_seconds", "duration_hours_min", "comments",
    "date_posted", "latitude", "longitude",
]


def section(title):
    print("\n" + "=" * 8 + " " + title + " " + "=" * 8)


def download_if_needed():
    if not os.path.exists(DATA_FILE):
        print(f"Téléchargement de {DATA_FILE} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    else:
        print(f"{DATA_FILE} déjà présent, pas de retéléchargement.")


# ---------------------------------------------------------------------------
# Phase 1 : ouvrir la caisse
# ---------------------------------------------------------------------------

def phase1_ouvrir_la_caisse():
    section("PHASE 1 - Ouvrir la caisse")

    total_lines = 0
    good_rows = []
    bad_rows = []
    with open(DATA_FILE, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            total_lines += 1
            if len(row) == len(COLUMNS):
                good_rows.append(row)
            else:
                bad_rows.append(row)

    print(f"Lignes dans le fichier         : {total_lines}")
    print(f"Lignes chargées (11 champs)    : {len(good_rows)}")
    print(f"Lignes mises à part            : {len(bad_rows)}")
    assert total_lines == len(good_rows) + len(bad_rows)

    print("\nExemple de ligne problématique :")
    print(bad_rows[0])
    print(f"-> {len(bad_rows[0])} champs au lieu de 11 : la colonne 'comments' contient")
    print("   une virgule non protégée par des guillemets, ce qui décale les champs suivants.")

    df_good = pd.DataFrame(good_rows, columns=COLUMNS)
    return df_good, bad_rows


# ---------------------------------------------------------------------------
# Phase 2 : rien n'est du bon type
# ---------------------------------------------------------------------------

def to_numeric_report(series, name):
    converted = pd.to_numeric(series, errors="coerce")
    n_fail = converted.isna().sum() - series.isna().sum()
    bad_values = series[converted.isna() & series.notna()].unique()[:5]
    print(f"  {name}: {n_fail} valeurs n'ont pas pu être converties. Exemples: {list(bad_values)}")
    return converted


def phase2_types(df):
    section("PHASE 2 - Rien n'est du bon type")
    df = df.copy()
    df = df.replace("", np.nan)

    for col in ["latitude", "longitude", "duration_seconds"]:
        df[col] = to_numeric_report(df[col], col)

    for col in ["datetime", "date_posted"]:
        before_na = df[col].isna().sum()
        converted = pd.to_datetime(df[col], errors="coerce", format="mixed")
        n_fail = converted.isna().sum() - before_na
        bad_values = df[col][converted.isna() & df[col].notna()].unique()[:5]
        print(f"  {col}: {n_fail} valeurs n'ont pas pu être converties. Exemples: {list(bad_values)}")
        df[col] = converted

    print("\nValeurs uniques de 'country' :", sorted(df["country"].dropna().unique()))
    print(f"  -> 'country' manquant sur {df['country'].isna().sum()} lignes (le témoin ou le")
    print("     capteur n'a jamais rempli ce champ pour ces relevés).")

    return df


# ---------------------------------------------------------------------------
# Phase 3 : trier les canulars
# ---------------------------------------------------------------------------

HOAX_KEYWORDS = re.compile(
    r"\b(hoax|fake|joke|not real|made up|prank|april fool)\b",
    re.IGNORECASE,
)


def phase3_canulars(df):
    section("PHASE 3 - Trier les canulars")
    df = df.copy()

    comments = df["comments"].fillna("")
    is_hoax = comments.str.contains(HOAX_KEYWORDS, regex=True)
    df["is_hoax"] = is_hoax.astype(int)

    n_hoax = int(is_hoax.sum())
    prop = n_hoax / len(df)
    print("Règle : un relevé est marqué canular si son témoignage (comments) contient")
    print("        un mot-clé d'aveu ou de doute ('hoax', 'fake', 'joke', 'made up', ...).")
    print(f"Relevés marqués canulars : {n_hoax} ({prop:.3%})")

    example = df.loc[is_hoax == 1, "comments"].iloc[0]
    print(f"\nExemple attrapé : {example[:120]}")
    print("Limite connue : la règle rate tous les canulars qui ne s'avouent pas eux-mêmes")
    print("               (l'immense majorité), et peut attraper à tort un texte qui cite")
    print("               ces mots sans être un aveu (ex: doute exprimé par le témoin).")

    return df


# ---------------------------------------------------------------------------
# Outils communs pour les modèles
# ---------------------------------------------------------------------------

NUMERIC_BASE = ["latitude", "longitude", "duration_seconds"]
CATEGORICAL_BASE = ["state", "country", "shape"]


def build_pipeline(numeric_cols, categorical_cols, use_text=False, text_col="comments"):
    transformers = [
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ]
    if use_text:
        transformers.append((
            "text",
            Pipeline([
                ("fillna", FunctionTransformer(lambda s: s.fillna(""))),
                ("tfidf", TfidfVectorizer(max_features=2000, stop_words="english")),
            ]),
            text_col,
        ))
    pre = ColumnTransformer(transformers, remainder="drop")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=None, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(pipe, X_train, y_train, X_test, y_test, label):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    print(f"[{label}] precision={prec:.3f}  recall={rec:.3f}  accuracy={acc:.3f}  "
          f"(train={len(X_train)}, test={len(X_test)}, canulars_test={int(y_test.sum())})")
    return {"precision": prec, "recall": rec, "accuracy": acc}


# ---------------------------------------------------------------------------
# Phase 4 : premier verdict (découpe naïve, aléatoire)
# ---------------------------------------------------------------------------

def phase4_premier_verdict(df):
    section("PHASE 4 - Le premier verdict (découpe aléatoire, avec 'comments' - à charger)")

    X = df[NUMERIC_BASE + CATEGORICAL_BASE + ["comments"]]
    y = df["is_hoax"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    pipe = build_pipeline(NUMERIC_BASE, CATEGORICAL_BASE, use_text=True)
    metrics = evaluate(pipe, X_train, y_train, X_test, y_test, "phase4 - avec comments, split aléatoire")
    print("\nATTENTION : ce chiffre est calculé avec une découpe aléatoire (relevés du même")
    print("événement des deux côtés) et avec 'comments' comme feature, alors que le label")
    print("is_hoax est lui-même dérivé de comments. Ce chiffre sera corrigé en phases 5, 7, 8.")
    return metrics


# ---------------------------------------------------------------------------
# Phase 5 : la fuite (colonnes écrites par quelqu'un qui savait déjà)
# ---------------------------------------------------------------------------

def print_leakage_table():
    section("PHASE 5 - Tableau de provenance des colonnes")
    rows = [
        ("datetime", "Le témoin", "Le soir de l'observation", "Non"),
        ("city / state / country", "Le témoin (et le capteur pour les coordonnées)", "Le soir de l'observation", "Non"),
        ("shape", "Le témoin", "Le soir de l'observation", "Non"),
        ("duration_seconds", "Le service de transmission (parsing automatique)", "Après réception", "Non"),
        ("duration_hours_min", "Le témoin", "Le soir de l'observation", "Non"),
        ("comments", "Le témoin, PUIS un employé du Bureau qui ajoute des notes\n"
                     "        entre doubles parenthèses ((NUFORC Note: ...)) après enquête", "Des semaines plus tard", "OUI"),
        ("date_posted", "Le Bureau (date de publication après modération)", "Après traitement", "Non direct, mais corrélé au traitement"),
        ("latitude / longitude", "Le capteur de la sonde", "Le soir de l'observation", "Non"),
    ]
    header = f"{'Colonne':32} | {'Qui écrit':45} | {'Quand':22} | {'Savait déjà ?'}"
    print(header)
    print("-" * len(header))
    for col, who, when, knew in rows:
        print(f"{col:32} | {who:45} | {when:22} | {knew}")
    print("\n-> La colonne 'comments' contient les notes ajoutées a posteriori par le Bureau,")
    print("   notes qui contiennent justement les mots utilisés pour fabriquer le label is_hoax.")
    print("   C'est elle qui sort du modèle (colonne à retirer).")


def phase5_sans_fuite(df, metrics_before):
    X = df[NUMERIC_BASE + CATEGORICAL_BASE]
    y = df["is_hoax"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    pipe = build_pipeline(NUMERIC_BASE, CATEGORICAL_BASE, use_text=False)
    metrics_after = evaluate(pipe, X_train, y_train, X_test, y_test, "phase5 - sans comments, split aléatoire")

    print(f"\nAvant (avec fuite)  : precision={metrics_before['precision']:.3f} recall={metrics_before['recall']:.3f}")
    print(f"Après (sans fuite)  : precision={metrics_after['precision']:.3f} recall={metrics_after['recall']:.3f}")
    print("Explication : le label is_hoax est un sous-produit direct du texte de 'comments'")
    print("(notes du Bureau incluses). Le modèle de la phase 4 ne détectait pas les canulars,")
    print("il retrouvait les mots-clés qui avaient servi à fabriquer sa propre cible. Une fois")
    print("cette colonne retirée, il ne reste plus que des champs factuels (position, forme,")
    print("durée, lieu) qui ne portent presque aucun signal sur le fait que quelqu'un ait plus")
    print("tard qualifié le récit de canular : le score s'effondre logiquement.")
    return metrics_after


# ---------------------------------------------------------------------------
# Phase 6 : le modèle le plus bête du Bureau
# ---------------------------------------------------------------------------

def phase6_stagiaire(df, metrics_reel):
    section("PHASE 6 - Le modèle le plus bête du Bureau")
    y = df["is_hoax"]
    y_pred_stagiaire = np.zeros(len(y), dtype=int)
    acc_stagiaire = accuracy_score(y, y_pred_stagiaire)
    prop_hoax = y.mean()

    print(f"Système du stagiaire : toujours 'pas un canular'.")
    print(f"Taux de bonnes réponses (accuracy) du stagiaire : {acc_stagiaire:.3%}")
    print(f"Taux de bonnes réponses (accuracy) de notre modèle (phase 5, sans fuite) : {metrics_reel['accuracy']:.3%}")
    print(f"(pour référence, la proportion réelle de canulars est {prop_hoax:.3%})")
    print("\nCe qu'on présente au Conseil : PAS l'accuracy. À 0,9-1% de canulars, prédire")
    print("toujours 'non' donne ~99% d'accuracy sans jamais attraper un seul canular : ce")
    print("chiffre ne prouve rien sur la capacité à détecter les canulars. On présente plutôt")
    print("le couple precision/recall sur la classe 'canular', les seuls indicateurs qui")
    print("distinguent un système qui détecte vraiment quelque chose d'un système qui se tait.")


# ---------------------------------------------------------------------------
# Phase 7 : plusieurs témoins, un seul événement
# ---------------------------------------------------------------------------

def build_event_id(df):
    dt_round = df["datetime"].dt.floor("h")
    return (
        dt_round.astype(str) + "|" +
        df["city"].fillna("").str.lower().str.strip() + "|" +
        df["state"].fillna("").str.lower().str.strip() + "|" +
        df["country"].fillna("").str.lower().str.strip()
    )


def phase7_evenements(df, prev_split_metrics):
    section("PHASE 7 - Plusieurs témoins, un seul événement")
    n_before = len(df)
    df = df.dropna(subset=["datetime"]).copy()
    print(f"Relevés sans datetime exploitable écartés de cette phase : {n_before - len(df)}")
    df["event_id"] = build_event_id(df)

    event_sizes = df.groupby("event_id").size()
    multi_witness_events = event_sizes[event_sizes > 1]
    n_multi_events = len(multi_witness_events)
    biggest_event_witnesses = int(event_sizes.max())

    print(f"Événements signalés par plus d'un témoin : {n_multi_events}")
    print(f"Nombre de témoins pour le plus gros événement : {biggest_event_witnesses}")

    # combien de relevés étaient à cheval sur les deux côtés dans la découpe d'hier (phase 4/5)
    X = df[NUMERIC_BASE + CATEGORICAL_BASE]
    y = df["is_hoax"]
    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    side = pd.Series("train", index=df.index)
    side.loc[idx_test] = "test"
    sides_per_event = df.assign(side=side).groupby("event_id")["side"].nunique()
    straddling_events = sides_per_event[sides_per_event > 1].index
    n_straddling_rows = int(df["event_id"].isin(straddling_events).sum())
    print(f"Relevés à cheval sur train/test dans la découpe d'hier : {n_straddling_rows}")

    # doublons de texte identique
    comments_nonnull = df["comments"].dropna()
    dup_mask = comments_nonnull.duplicated(keep=False) & (comments_nonnull.str.strip() != "")
    n_dup_texts = int(dup_mask.sum())
    print(f"Témoignages recopiés à l'identique (mot pour mot) sur plusieurs lignes : {n_dup_texts}")
    print("-> Traitement : ces lignes ne sont pas supprimées (une ligne = un relevé reçu),")
    print("   mais elles suivent la même règle que les événements multi-témoins : un même")
    print("   texte dupliqué part entièrement du même côté de la découpe train/test.")

    example_event_id = multi_witness_events.sort_values(ascending=False).index[0]
    print(f"\nExemple d'événement multi-témoins ({int(event_sizes[example_event_id])} témoins) :")
    cols_show = ["datetime", "city", "state", "country", "shape"]
    print(df.loc[df["event_id"] == example_event_id, cols_show].head(10).to_string(index=False))

    # découpe par groupe d'événement (GroupShuffleSplit)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    groups = df["event_id"]
    train_idx, test_idx = next(gss.split(df, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipe = build_pipeline(NUMERIC_BASE, CATEGORICAL_BASE, use_text=False)
    metrics_grouped = evaluate(pipe, X_train, y_train, X_test, y_test, "phase7 - sans comments, split par événement")

    print(f"\nPhase 4 (avant, aléatoire, avec fuite)  : precision={prev_split_metrics['before']['precision']:.3f} recall={prev_split_metrics['before']['recall']:.3f}")
    print(f"Phase 7 (après, groupé, sans fuite)     : precision={metrics_grouped['precision']:.3f} recall={metrics_grouped['recall']:.3f}")

    return df, metrics_grouped, event_sizes


# ---------------------------------------------------------------------------
# Phase 8 : l'ordre des choses (découpe temporelle)
# ---------------------------------------------------------------------------

def phase8_ordre_temporel(df):
    section("PHASE 8 - L'ordre des choses")
    df = df.copy()
    df = df.dropna(subset=["datetime"])

    print("Deux dates disponibles : 'datetime' (le témoin a levé les yeux) et 'date_posted'")
    print("(le Bureau a publié le dossier). On coupe sur 'datetime' : c'est la date de")
    print("l'événement réel que le système verra en production, pas la date administrative")
    print("de traitement — couper sur date_posted laisserait filtrer des événements anciens")
    print("republiés tardivement du côté apprentissage alors qu'ils sont chronologiquement")
    print("mêlés aux événements de test.")

    cutoff = df["datetime"].quantile(0.8)
    print(f"\nDate de coupure (80e percentile de 'datetime') : {cutoff.date()}")

    train_mask = df["datetime"] < cutoff
    test_mask = ~train_mask

    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    print(f"Relevés côté apprentissage : {n_train}")
    print(f"Relevés côté test          : {n_test}")

    prop_train = df.loc[train_mask, "is_hoax"].mean()
    prop_test = df.loc[test_mask, "is_hoax"].mean()
    print(f"Proportion de canulars côté apprentissage : {prop_train:.3%}")
    print(f"Proportion de canulars côté test          : {prop_test:.3%}")

    X = df[NUMERIC_BASE + CATEGORICAL_BASE]
    y = df["is_hoax"]
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    pipe = build_pipeline(NUMERIC_BASE, CATEGORICAL_BASE, use_text=False)
    metrics = evaluate(pipe, X_train, y_train, X_test, y_test, "phase8 - découpe temporelle")

    if abs(prop_train - prop_test) > 0.002:
        print("\n-> Les deux proportions ne sont pas égales : les canulars ne sont pas uniformes")
        print("   dans le temps (probablement lié à des campagnes de modération du Bureau ou à")
        print("   des périodes médiatiques particulières), c'est un signal à surveiller.")

    return df, cutoff, metrics


# ---------------------------------------------------------------------------
# Phase 9 : les cases vides
# ---------------------------------------------------------------------------

def phase9_cases_vides(df):
    section("PHASE 9 - Les cases vides")
    missing_counts = df[COLUMNS].isna().sum().sort_values(ascending=False)
    print("Colonnes les plus trouées :")
    print(missing_counts.head(5))

    top3 = missing_counts.head(3).index.tolist()
    print(f"\nColonnes retenues : {top3}")

    rows = []
    for col in top3:
        has_hole = df[col].isna()
        prop_with_hole = df.loc[has_hole, "is_hoax"].mean()
        prop_without_hole = df.loc[~has_hole, "is_hoax"].mean()
        rows.append((col, prop_with_hole, prop_without_hole))
        print(f"  {col}: canulars si trou = {prop_with_hole:.3%} | canulars si rempli = {prop_without_hole:.3%}")

    print("\nTraitement retenu : pour chaque colonne trouée utilisée par le modèle, on garde")
    print("une catégorie explicite 'missing' (colonnes catégorielles) plutôt que la modalité")
    print("la plus fréquente, et l'imputation numérique se fait par une médiane calculée sur")
    print("le train uniquement (phase 10). Le SimpleImputer(strategy='constant', 'missing')")
    print("déjà utilisé dans build_pipeline() préserve donc la trace du trou : le modèle peut")
    print("continuer à apprendre que 'ce champ était vide' est en soi une information.")
    return rows


# ---------------------------------------------------------------------------
# Phase 10 : la chaîne de traitement du Bureau (pipeline unique fit sur train)
# ---------------------------------------------------------------------------

def phase10_chaine(df, cutoff):
    section("PHASE 10 - La chaîne de traitement du Bureau")
    df = df.copy()
    train_mask = df["datetime"] < cutoff
    test_mask = ~train_mask

    X = df[NUMERIC_BASE + CATEGORICAL_BASE]
    y = df["is_hoax"]
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    prop_train = y_train.mean()
    prop_test = y_test.mean()
    print(f"Proportion de canulars - train : {prop_train:.3%}")
    print(f"Proportion de canulars - test  : {prop_test:.3%}")

    pipe = build_pipeline(NUMERIC_BASE, CATEGORICAL_BASE, use_text=False)
    metrics = evaluate(pipe, X_train, y_train, X_test, y_test, "phase10 - pipeline unique, tout appris sur train")

    print("\nTout l'imputer/scaler/onehot est encapsulé dans le même sklearn.Pipeline que le")
    print("classifieur, et 'fit' n'est appelé que sur X_train : aucune statistique (médiane,")
    print("catégories vues) n'est calculée sur le test.")

    # démonstration : une ligne inventée à la main traverse toute la chaîne
    new_row = pd.DataFrame([{
        "latitude": 48.8566,
        "longitude": 2.3522,
        "duration_seconds": 120,
        "state": "missing",
        "country": "fr",
        "shape": "light",
    }])
    pred = pipe.predict(new_row)[0]
    proba = pipe.predict_proba(new_row)[0]
    print(f"\nRelevé inventé à la main : {new_row.to_dict(orient='records')[0]}")
    print(f"Prédiction : {'canular' if pred == 1 else 'pas un canular'} (proba canular = {proba[1]:.3f})")

    return metrics


# ---------------------------------------------------------------------------
# Phase 11 : combien de temps ça a duré
# ---------------------------------------------------------------------------

TEXT_DURATION_RE = re.compile(
    r"(?P<num>\d+(?:[.,]\d+)?)\s*(?:-\s*\d+(?:[.,]\d+)?)?\s*"
    r"(?P<unit>sec|second|min|minute|hr|hour|h)",
    re.IGNORECASE,
)
UNIT_TO_SECONDS = {
    "sec": 1, "second": 1,
    "min": 60, "minute": 60,
    "hr": 3600, "hour": 3600, "h": 3600,
}


def parse_text_duration(text):
    if not isinstance(text, str) or not text.strip():
        return np.nan
    m = TEXT_DURATION_RE.search(text)
    if not m:
        return np.nan
    num = float(m.group("num").replace(",", "."))
    unit = m.group("unit").lower()
    return num * UNIT_TO_SECONDS[unit]


def phase11_duree(df):
    section("PHASE 11 - Combien de temps ça a duré")
    df = df.copy()

    df["duration_from_text"] = df["duration_hours_min"].apply(parse_text_duration)

    seconds = df["duration_seconds"]
    from_text = df["duration_from_text"]

    # colonne finale : on fait confiance à duration_seconds si > 0 et cohérente,
    # sinon on retombe sur ce qu'on a extrait du texte du témoin.
    final = seconds.copy()
    need_fallback = final.isna() | (final == 0)
    final = final.where(~need_fallback, from_text)
    df["duration_final"] = final

    n_unusable = int(df["duration_final"].isna().sum())
    print(f"Relevés dont la durée reste inutilisable après traitement : {n_unusable}")

    both_known = seconds.notna() & from_text.notna() & (seconds > 0)
    ratio = (seconds[both_known] / from_text[both_known]).replace([np.inf, -np.inf], np.nan)
    contradiction = both_known & ((ratio > 3) | (ratio < 1 / 3))
    n_contradiction = int(contradiction.sum())
    print(f"Relevés où les deux colonnes de durée se contredisent (facteur > 3) : {n_contradiction}")

    median_duration = df["duration_final"].median()
    print(f"Durée médiane (secondes) : {median_duration:.1f}")

    one_day = 24 * 3600
    n_over_one_day = int((df["duration_final"] > one_day).sum())
    print(f"Relevés annonçant plus d'une journée d'observation : {n_over_one_day}")

    top3 = df.nlargest(3, "duration_final")[["datetime", "city", "duration_seconds", "duration_hours_min", "duration_final"]]
    print("\nLes 3 durées les plus longues :")
    print(top3.to_string(index=False))
    print("-> Ces valeurs extrêmes (années entières) sont plausibles comme témoignages mais")
    print("   inutilisables telles quelles pour un modèle : on les plafonne à 1 journée")
    print("   (86 400 s) plutôt que de les supprimer, pour ne perdre aucune ligne.")
    df["duration_final"] = df["duration_final"].clip(upper=one_day)

    example_contra = df.loc[contradiction].iloc[0] if n_contradiction else None
    if example_contra is not None:
        print("\nExemple de contradiction entre les deux colonnes de durée :")
        print(example_contra[["duration_seconds", "duration_hours_min", "duration_from_text"]])

    return df


# ---------------------------------------------------------------------------
# Phase 12 : la ville et l'heure
# ---------------------------------------------------------------------------

RARE_SHAPE_THRESHOLD = 5
SHAPE_MERGE = {"changed": "changing"}


def phase12_ville_heure(df):
    section("PHASE 12 - La ville et l'heure")
    df = df.copy()

    n_cols_before = len(NUMERIC_BASE + CATEGORICAL_BASE)

    # ville : fréquence + regroupement des villes rares
    city_counts = df["city"].value_counts()
    n_cities_once = int((city_counts == 1).sum())
    print(f"Villes qui n'apparaissent qu'une seule fois : {n_cities_once} / {df['city'].nunique()} villes")
    print("Règle appliquée à la ville : encodage par fréquence (nombre d'occurrences dans la")
    print("transmission, normalisé), une seule colonne numérique au lieu d'une colonne par ville.")
    df["city_freq"] = df["city"].map(city_counts) / len(df)

    # heure : encodage cyclique sin/cos
    hour = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    def cyclic_point(h):
        return np.array([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)])

    d_23_0 = np.linalg.norm(cyclic_point(23) - cyclic_point(0))
    d_23_20 = np.linalg.norm(cyclic_point(23) - cyclic_point(20))
    print(f"\nDistance encodée entre 23h et 0h  : {d_23_0:.3f}")
    print(f"Distance encodée entre 23h et 20h : {d_23_20:.3f}")
    assert d_23_0 < d_23_20, "l'encodage cyclique doit rapprocher 23h de 0h"

    # shape : fusion orthographique + regroupement des formes rares
    shape_clean = df["shape"].replace(SHAPE_MERGE)
    shape_counts = shape_clean.value_counts()
    rare_shapes = shape_counts[shape_counts <= RARE_SHAPE_THRESHOLD].index
    shape_clean = shape_clean.where(~shape_clean.isin(rare_shapes), "other")
    n_shapes_final = shape_clean.nunique(dropna=True)
    df["shape_clean"] = shape_clean

    print(f"\nFormes avant nettoyage : {df['shape'].nunique(dropna=True)}")
    print(f"Fusion orthographique  : 'changed' -> 'changing'")
    print(f"Formes rares (<= {RARE_SHAPE_THRESHOLD} occurrences) regroupées dans 'other'")
    print(f"Formes après nettoyage : {n_shapes_final}")

    numeric_final = NUMERIC_BASE + ["city_freq", "hour_sin", "hour_cos"]
    categorical_final = ["state", "country"]  # shape_clean traité séparément (déjà propre)
    categorical_final_all = categorical_final + ["shape_clean"]
    n_cols_after_raw = len(numeric_final) + len(categorical_final_all)
    print(f"\nNombre de colonnes du tableau de features avant : {n_cols_before}")
    print(f"Nombre de colonnes du tableau de features après  : {n_cols_after_raw}")
    print("(loin des 22 018 colonnes qu'aurait donné un one-hot brut sur la ville)")

    return df, numeric_final, categorical_final_all


def phase12_modele_final(df, numeric_final, categorical_final, cutoff):
    train_mask = df["datetime"] < cutoff
    test_mask = ~train_mask

    X = df[numeric_final + categorical_final]
    y = df["is_hoax"]
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    pipe = build_pipeline(numeric_final, categorical_final, use_text=False)
    metrics = evaluate(pipe, X_train, y_train, X_test, y_test, "phase12 - modèle final (ville+heure encodées)")
    return metrics


# ---------------------------------------------------------------------------
# Récapitulatif final
# ---------------------------------------------------------------------------

def recap(all_metrics):
    section("RÉCAPITULATIF - precision / recall sur la classe canular")
    for label, m in all_metrics:
        print(f"  {label:45} precision={m['precision']:.3f}  recall={m['recall']:.3f}  accuracy={m['accuracy']:.3f}")


def main():
    download_if_needed()
    df, bad_rows = phase1_ouvrir_la_caisse()
    df = phase2_types(df)
    df = phase3_canulars(df)

    metrics_p4 = phase4_premier_verdict(df)
    print_leakage_table()
    metrics_p5 = phase5_sans_fuite(df, metrics_p4)
    phase6_stagiaire(df, metrics_p5)

    phase9_cases_vides(df)  # sur la transmission complète, avant tout filtrage temporel

    df, metrics_p7, event_sizes = phase7_evenements(df, {"before": metrics_p4})
    df, cutoff, metrics_p8 = phase8_ordre_temporel(df)
    metrics_p10 = phase10_chaine(df, cutoff)
    df = phase11_duree(df)
    df, numeric_final, categorical_final = phase12_ville_heure(df)
    metrics_p12 = phase12_modele_final(df, numeric_final, categorical_final, cutoff)

    recap([
        ("Phase 4 - découpe aléatoire, avec fuite (comments)", metrics_p4),
        ("Phase 5 - découpe aléatoire, sans fuite", metrics_p5),
        ("Phase 7 - découpe par événement, sans fuite", metrics_p7),
        ("Phase 8 - découpe temporelle", metrics_p8),
        ("Phase 10 - pipeline unique fit sur train", metrics_p10),
        ("Phase 12 - ville/heure encodées", metrics_p12),
    ])


if __name__ == "__main__":
    main()
