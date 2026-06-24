import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def carregar_dados():
    df = pd.read_csv("dados_chagas.csv")
    df.set_index(df.columns[0], inplace=True)
    return df


def preprocessar(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_teste = scaler.transform(X_teste)

    return X_treino, X_teste, y_treino, y_teste