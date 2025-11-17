# Solar Radiation Forecasting – End-to-End ML & Orchestration

Pipeline de **régression horaire** pour prédire la production solaire (irradiation / GHI) à partir de séries temporelles météorologiques, avec :

- entraînement et évaluation de modèles de ML,
- orchestration des tâches avec **Apache Airflow** via **Astro CLI**,
- API de prédiction avec **FastAPI**,
- dashboard de visualisation avec **Streamlit**,
- conteneurisation avec **Docker**.

---

## 🎯 Objectif

Fournir des prévisions fiables de production solaire à l’échelle horaire pour aider à :

- optimiser la **gestion énergétique**,
- anticiper la production photovoltaïque,
- tester différents modèles et stratégies de déploiement dans un cadre MLOps.

---

## 🧱 Fonctionnalités principales

- Ingestion et parsing de séries temporelles (horodatage, nettoyage, tri).
- Feature engineering temporel (heures, mois, jour, encodage cyclique, etc.).
- Détection et traitement des **outliers**, split temporel **chronologique** train/test.
- Benchmark de modèles :
  - `LinearRegression`
  - `RandomForestRegressor`
  - `XGBRegressor`
  - `GradientBoostingRegressor`
- Tuning (GridSearch / RandomSearch) et évaluation avec :
  - R²
  - MAE
  - MSE
- Orchestration des pipelines via **Airflow** (Astro CLI).
- Exposition d’un endpoint de prédiction via **FastAPI**.
- Dashboard de visualisation / monitoring via **Streamlit**.
- Conteneurisation et exécution via **Docker** / `docker-compose`.

---

## 📊 Résultats (exemple)

Meilleur modèle trouvé sur la base des expérimentations :

- **Modèle** : `GradientBoostingRegressor`
- **r²_test** : 0.707  
- **MAE** : 49.07  
- **MSE** : 4 445.72  

---

## 🧰 Stack technique

### Langage & Data

- Python
- NumPy
- pandas

### Machine Learning

- scikit-learn
- XGBoost
- (optionnel) LightGBM

### Orchestration & MLOps

- Apache Airflow
- **Astro CLI** (Astro Runtime)
- Docker

> 📝 Astro Runtime inclut déjà de nombreux *providers* Airflow pré‑installés : voir la doc officielle  
> (section *Astro Runtime – provider packages*).

### API & Front

- FastAPI
- Streamlit

### Autres bibliothèques

- matplotlib
- seaborn
- plotly
- joblib
- requests
- python-dotenv
- psycopg2-binary (PostgreSQL)
- boto3 / botocore / aiobotocore (intégration AWS S3, asynchrone si besoin)
- protobuf (compatibilité avec certains frameworks ML)

---

## 🗂️ Structure (simplifiée)

```text
.
├── dags/
│   ├── process_weather.py      # DAG d'entraînement et préparation des données
│   └── wheatheretl.py          # DAG ETL + inférence / insertion en base
├── src/
│   ├── functions.py            # Fonctions utilitaires (ETL, features, modèles, S3, etc.)
│   └── inference.py            # Classe / fonctions d'inférence
├── app/
│   └── main.py                 # App Streamlit ou FastAPI (selon organisation)
├── notebooks/
│   └── solar_radiation_predictions_ML.ipynb   # Notebook d'exploration / prototypage
├── Dockerfile                  # Image principale
├── docker-compose.yml          # Orchestration locale (Airflow, API, Streamlit, DB...)
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

> La structure exacte peut évoluer, mais cette vue donne les grandes briques du projet.

---

## 🔧 Installation globale (sans Astro CLI)

### 1. Cloner le dépôt

```bash
git clone https://github.com/Thibaut-Longchamps/solar-radiation-forecasting.git
cd solar-radiation-forecasting
```

### 2. Créer un environnement virtuel (optionnel mais recommandé)

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# ou
.venv\Scripts\activate       # Windows
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Orchestration avec Airflow & Astro CLI

Cette partie décrit l’utilisation de **Apache Airflow** via **Astro CLI** pour orchestrer les DAGs de :

- préparation & entraînement des modèles (`process_weather.py`),
- ETL météo + inférence + insertion en base (`wheatheretl.py`).

### 1️⃣ Prérequis

- **Docker** installé et fonctionnel.
- **Astro CLI** installé.
- **AWS S3**

📎 Installation Astro CLI (résumé, voir doc Astronomer pour les détails) :

- **macOS (Homebrew)**  
  ```bash
  brew install astro
  ```

- **Linux**  
  ```bash
  curl -sSL https://install.astronomer.io | sudo bash
  ```

- **Windows**  
  - Recommandé : passer par **WSL2** et utiliser la commande Linux ci‑dessus.

### 2️⃣ Initialisation (si projet Astro non encore créé)

Si ton repo n’est pas encore initialisé en projet Astro :

```bash
astro dev init
```

Cela crée la structure standard `.astro/`, les fichiers de base Airflow, etc.  
Tu peux ensuite déplacer tes DAGs dans `dags/` et ton code dans `src/`.

### 3️⃣ Dépendances Airflow (requirements)

Dans le projet Astro, assure-toi que `requirements.txt` contient au moins :

```txt
numpy==1.26.4
pandas
scikit-learn==1.5.2
xgboost==2.1.2
lightgbm
joblib==1.4.2
matplotlib==3.9.2
seaborn==0.13.2
plotly==5.24.1
boto3
botocore
aiobotocore
psycopg2-binary==2.9.10
python-dotenv==1.0.0
requests==2.32.3
protobuf==3.20.0
```

Astro Runtime se chargera d’installer ces paquets dans l’image Airflow au démarrage.

### 4️⃣ Lancer l’environnement Airflow local

```bash
astro dev start
```

Cela va :

- builder l’image Docker Airflow avec tes dépendances,
- lancer les containers Airflow (webserver, scheduler, DB, etc.).

Ensuite, tu peux accéder à l’UI Airflow :

- `http://localhost:8080`

### 5️⃣ Configurer les connexions Airflow

Dans l’UI : **Admin → Connections**.

#### Connexion PostgreSQL – `postgres_default`

- Conn Id : `postgres_default`
- Conn Type : **Postgres**
- Host : selon ton `docker-compose` (ex : `postgres`)
- Port : `5432` ou mappage local
- Login / Password : ex. `postgres` / `postgres`
- Database : `postgres`

#### Connexion HTTP – `open_meteo_api`

- Conn Id : `open_meteo_api`
- Conn Type : **HTTP**
- Host : `https://api.open-meteo.com/`

Utilisée pour récupérer les données météo (historiques ou temps réel).

#### Connexion AWS S3 – `aws_s3_conn` (optionnel)

- Conn Id : `aws_s3_conn`
- Conn Type : **Amazon Web Services**
- Extra JSON, ex. :

```json
{
  "aws_access_key_id": "XXXXX",
  "aws_secret_access_key": "YYYYY",
  "region_name": "eu-west-3"
}
```

Utilisée pour stocker les modèles, prédictions ou jeux de données sur S3.

### 6️⃣ Activer les DAGs

Depuis l’interface Airflow :

- Activer le DAG d’entraînement / préparation (`process_weather.py`).
- Activer le DAG ETL météo + inférence (`wheatheretl.py`).

Surveiller l’exécution (onglets *Grid* / *Graph*).

---

## 🧪 API FastAPI & Dashboard Streamlit

Selon l’organisation du projet, tu peux avoir :

### 🔹 API FastAPI (exemple)

Démarrer l’API :

```bash
uvicorn main:app --reload
```

- Endpoint exemple : `GET /predict` ou `POST /predict`
- Entrée : features météo / temporelles
- Sortie : prédiction de GHI / production solaire.

### 🔹 Streamlit

Démarrer le dashboard :

```bash
streamlit run main.py
```

- Visualisation de la série temporelle de GHI
- Possibilité de comparer modèles, plages temporelles, etc.

Assure-toi que la chaîne de connexion PostgreSQL dans le code Streamlit est cohérente avec ton environnement (host, port, user, password).

---

## 🐳 Docker & docker-compose

### Build de l’image

```bash
docker build -t solar-radiation-forecasting .
```

### Démarrage via docker-compose

```bash
docker-compose up
```

En fonction de la configuration de `docker-compose.yml`, cela peut lancer :

- Airflow (webserver, scheduler, DB, etc.),
- l’API FastAPI,
- Streamlit,
- PostgreSQL.

---

## 🤝 Contributions

Les contributions sont les bienvenues.

1. Forker le dépôt.
2. Créer une branche de feature :  
   ```bash
   git checkout -b feature/ma-feature
   ```
3. Committer vos modifications :  
   ```bash
   git commit -m "Ajout nouvelle fonctionnalité"
   ```
4. Pousser la branche :  
   ```bash
   git push origin feature/ma-feature
   ```
5. Ouvrir une **Pull Request**.

---

## 📜 Licence

Ce projet peut être distribué sous une licence open-source (par exemple MIT).  
Adapter la section selon le fichier `LICENSE` présent dans le dépôt.
