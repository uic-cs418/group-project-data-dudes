import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ML 1
def get_classification_results(df):
    target = 'MEDBIAS_W119'
    features = [
        'AIHCCOMF_W119','AI_HEARD_W119','USEAI_W119','F_AGECAT',
        'F_GENDER','F_RACETHNMOD','F_EDUCCAT','F_INC_TIER2',
        'F_METRO','F_PARTY_FINAL','EMPLSIT_W119','AIHCTRT1_W119',
        'HIREBIAS1_W119','DESRISK_COMF_W119','DESRISK_CREAT_W119',
        'DESRISK_NTECH_W119','RISK2_W119'
    ]

    df_clf = df[features + [target]].copy()

    cols_with_99 = [
        'F_EDUCCAT','F_INC_TIER2','AIHCTRT1_W119','HIREBIAS1_W119',
        'DESRISK_COMF_W119','DESRISK_CREAT_W119','DESRISK_NTECH_W119',
        'USEAI_W119','RISK2_W119','F_RACETHNMOD','F_PARTY_FINAL',
        'EMPLSIT_W119'
    ]
    for col in cols_with_99:
        df_clf[col] = df_clf[col].replace(99.0, np.nan)

    df_clf.dropna(subset=[target] + features, inplace=True)

    comfort_map = {
        "Very uncomfortable": 0,
        "Somewhat uncomfortable": 1,
        "Somewhat comfortable":   2,
        "Very comfortable":      3
    }
    heard_map = {
        "Nothing":  0,
        "A little": 1,
        "A lot":    2
    }
    df_clf['AIHCCOMF_W119'] = df_clf['AIHCCOMF_W119'].map(comfort_map)
    df_clf['AI_HEARD_W119'] = df_clf['AI_HEARD_W119'].map(heard_map)
    df_clf['RISK2_W119']    = df_clf['RISK2_W119'].map({1.0: 0, 2.0: 1})
    df_clf['F_METRO']       = df_clf['F_METRO'].map({1.0: 1, 2.0: 0})

    df_clf = pd.get_dummies(
        df_clf,
        columns=[
            'F_AGECAT','F_GENDER','F_RACETHNMOD',
            'F_PARTY_FINAL','EMPLSIT_W119'
        ],
        drop_first=True
    )

    X = df_clf.drop(columns=[target])
    y = df_clf[target].map({
        "Not a problem":   0,
        "Minor problem":   1,
        "Major problem":   2
    }).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results_df = pd.DataFrame({
        'Actual':    y_test.values,
        'Predicted': y_pred
    })
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Not a problem", "Minor problem", "Major problem"]

    return accuracy, results_df, cm, labels

# ML 2
def preprocess(df):
    features = [
        "AIHCCOMF_W119","AI_HEARD_W119","USEAI_W119","F_AGECAT","F_GENDER",
        "F_RACETHNMOD","F_EDUCCAT","F_INC_TIER2","F_METRO","F_PARTY_FINAL",
        "EMPLSIT_W119","AIHCTRT1_W119","HIREBIAS1_W119","DESRISK_COMF_W119",
        "DESRISK_CREAT_W119","DESRISK_NTECH_W119","RISK2_W119"
    ]
    dfc = df[features].copy()
    to_clean = [
        'F_EDUCCAT','F_INC_TIER2','AIHCTRT1_W119','HIREBIAS1_W119',
        'DESRISK_COMF_W119','DESRISK_CREAT_W119','DESRISK_NTECH_W119',
        'USEAI_W119','RISK2_W119','F_RACETHNMOD','F_PARTY_FINAL','EMPLSIT_W119'
    ]
    for col in to_clean:
        dfc[col] = dfc[col].replace(99.0, np.nan)
        dfc.dropna(inplace=True)

    dfc['AIHCCOMF_W119'] = dfc['AIHCCOMF_W119'].map({
        "Very uncomfortable": 0,
        "Somewhat uncomfortable": 1,
        "Somewhat comfortable": 2,
        "Very comfortable": 3
    })
    dfc['AI_HEARD_W119'] = dfc['AI_HEARD_W119'].map({
        "Nothing": 0, "A little": 1, "A lot": 2
    })
    dfc['RISK2_W119'] = dfc['RISK2_W119'].map({1.0: 0, 2.0: 1})
    dfc['F_METRO']    = dfc['F_METRO'].map({1.0: 1, 2.0: 0})

    dfc = pd.get_dummies(
        dfc,
        columns=['F_AGECAT','F_GENDER','F_RACETHNMOD','F_PARTY_FINAL','EMPLSIT_W119'],
        drop_first=True
    )

    X = dfc.values
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled

def cluster_and_embed(X, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_
    sizes = np.bincount(labels)

    pca = PCA(n_components=2).fit(X)
    X2 = pca.transform(X)
    centers2 = pca.transform(kmeans.cluster_centers_)
    return labels, sizes, X2, centers2

def plot_clusters(X2, labels, centers2, k=3):
    plt.figure(figsize=(6,5))
    plt.scatter(X2[:,0], X2[:,1], c=labels, cmap='viridis', alpha=0.7)
    plt.scatter(
        centers2[:,0], centers2[:,1],
        c='red', marker='X', s=100, edgecolors='k', label='Centroids'
    )
    plt.title(f'K‑Means Clusters (k={k}) in PCA Space')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    plt.show()
