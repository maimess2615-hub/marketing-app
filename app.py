import streamlit as st 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor


st.set_page_config(page_title="Dashboard Client",layout="wide")

st.title("📊Dasboard Marketing")
st.markdown("Analyse,segmentation et prediction client en temps reel")

#Upload fichier 
st.sidebar.header("📁charger vos donnees")
uploaded_file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["anciennete_client"] = df["anciennete_client"].apply(
    lambda x: np.nan if x < 0 else x
    )
    df["anciennete_client"] = df["anciennete_client"].fillna(
    df["anciennete_client"].median()
    )
    return df

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["revenu_annuel"] = df["annuel"].apply(
    lambda x: np.nan if x < 0 else x
    )
    df["revenu_annuel"] = df["revenu_annuel"].fillna(
    df["revenu_annuel"].median()
    )
    return df

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["score_fidelite"] = df["score_fidelite"].apply(
    lambda x: np.nan if x < 0 else x
    )
    df["score_fidelite"] = df["score_fidelite"].fillna(
    df["score_fidelite"].median()
    )
    return df

if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.warning("Veuillez importer un fichier CSV pour commencer")
    st.stop()
    
#KPI
st.subheader("📌Indicateurs cles (KPI)")
col1, col2, col3 = st.columns(3)
col1.metric("Revenu moyen", f"{df['revenu_annuel'].mean():,.0f}")   
col2.metric("Depenses moyennes", f"{df['depenses_moyennes'].mean():,.0f}")  
col3.metric("frequence moyenne", f"{df['frequence_achat'].mean():,.0f}")   

#Inputs
st.sidebar.header("🔧Parametres Client")
frequence = st.sidebar.slider("Frequnce d'achat", 1, 50, 10)
depenses = st.sidebar.slider("Depenses moyennes", 100, 10000, 1000)
anciennete = st.sidebar.slider("Anciennete client", 1, 20, 5)
age = st.sidebar.slider("Age du client", 18, 80, 30)

#Choix du nombre de clusters
k_value = st.sidebar.slider("Nombre de clusters(K)", 2, 6, 3)

#Donnees
X = df[["frequence_achat", "depenses_moyennes", "anciennete_client"]]
y = df["revenu_annuel"]

#Modele
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
model.fit(X_train, y_train)

#Score modele
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📊Performance du modele")
st.write(f"R2 : {r2:.2f}")
st.write(f"RMSE : {rmse:.2f}")

#Prediction
input_data = np.array([[frequence, depenses, anciennete]])
prediction = model.predict(input_data)

st.subheader("💵Prediction")
st.success(f"Revenu estime : {prediction[0]:,.2f}")

#K-means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k_value, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)
cluster = kmeans.predict(scaler.transform(input_data))[0]

#Intrpretation
labels = ["Faible valeur", "Medium", "Premium"]
reco_list = ["Promo", "Fidelisation", "Offres VIP"]
interpretation = labels[cluster % 3]
reco = reco_list[cluster % 3]

st.subheader("🎯segment client")
st.info(f"cluster {cluster} → {interpretation}")

st.subheader("👩🏾‍💻Recommandation du client")
st.warning(reco)

#Profil du client
st.subheader("👨🏾‍⚖️Profil du client")
profil = f"Client avec frequece {frequence},    depenses {depenses},   anciennete {anciennete}, age{age} ans.Segment : {interpretation}.Revenu estime : {prediction[0]:,.2f}"
st.write(profil)

#Storytelling automatique
st.subheader("📖Storytelling")
st.write(f"Ce client ayant {age}ans appartient a un segmet a {interpretation.lower()}, avec un potentiel de revenu estime a {prediction[0]:,.0f}. Il est recommande d'appliquer une strategie : {reco.lower()}.")

#Methode du coude
st.subheader("📊Methode du coude")
inertia = []
k_range = range(1, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(k_range, inertia, marker='o')
st.pyplot(fig_elbow)

#Correlation
st.subheader("📊Matrice de correlation")
fig_corr, ax_corr = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

#Filtre
st.subheader("fitrer par cluster")
selectected_cluster = st.selectbox("Choisir un cluster", sorted(df["cluster"].unique()))
st.dataframe(df[df["cluster"] == selectected_cluster])

#Visualisation
st.subheader("📈Segmentation des clients")
fig, ax = plt.subplots()
sns.scatterplot(x="revenu_annuel", y="depenses_moyennes", hue="cluster", palette="Set2", data=df, ax=ax)
st.pyplot(fig)

#Comparaison
st.subheader("📊Comparaison des segments")
segment_stats = df.groupby("cluster").mean(numeric_only=True)
st.dataframe(segment_stats)

fig3, ax3 = plt.subplots()
segment_stats[["revenu_annuel", "depenses_moyennes"]].plot(kind="bar", ax=ax3)
st.pyplot(fig3)

# Export
st.subheader("📥Export des resultats")
df_export = df.copy()
df_export["prediction_revenu"] = model.predict(X)

@st.cache_data
def convert_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_csv(df_export)
st.download_button("Telecharger les resultats", csv, "resultats.csv", "text/csv")

#Apercu
st.subheader("📁Apercu des donnees")
st.dataframe(df.head())

st.markdown("---")
st.markdown("Dashboard avance avec analyse, ML et aide a la decision")
