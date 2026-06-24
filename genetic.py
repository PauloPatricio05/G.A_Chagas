import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score


def criar_individuo(n_features):
    return np.random.randint(0, 2, n_features)


def criar_populacao(n_features, pop_size):
    return np.array([criar_individuo(n_features) for _ in range(pop_size)])


def fitness(ind, X_tr, y_tr, X_val, y_val, penalizacao):
    if np.sum(ind) == 0:
        return -1

    X_sel = X_tr[:, ind == 1]
    X_val_sel = X_val[:, ind == 1]

    model = GaussianNB()
    model.fit(X_sel, y_tr)

    pred = model.predict(X_val_sel)

    f1 = f1_score(y_val, pred, average="macro")

    return f1 - penalizacao * np.sum(ind)


def selecao(pop, scores, tournament_size):
    idx = np.random.choice(len(pop), tournament_size)
    best = idx[np.argmax(scores[idx])]
    return pop[best]


def crossover(p1, p2):
    ponto = np.random.randint(1, len(p1))
    return np.concatenate([p1[:ponto], p2[ponto:]])


def mutacao(ind, mutation_rate):
    for i in range(len(ind)):
        if np.random.rand() < mutation_rate:
            ind[i] = 1 - ind[i]
    return ind


def algoritmo_genetico(X_tr, y_tr, X_val, y_val, config):

    pop_size = config["pop_size"]
    generations = config["generations"]
    mutation_rate = config["mutation_rate"]
    tournament_size = config["tournament_size"]
    penalizacao = config["penalizacao"]

    n_features = X_tr.shape[1]

    pop = criar_populacao(n_features, pop_size)

    best_ind = None
    best_score = -np.inf

    for _ in range(generations):

        scores = np.array([
            fitness(ind, X_tr, y_tr, X_val, y_val, penalizacao)
            for ind in pop
        ])

        if np.max(scores) > best_score:
            best_score = np.max(scores)
            best_ind = pop[np.argmax(scores)].copy()

        new_pop = []

        for _ in range(pop_size):
            p1 = selecao(pop, scores, tournament_size)
            p2 = selecao(pop, scores, tournament_size)

            child = crossover(p1, p2)
            child = mutacao(child, mutation_rate)

            new_pop.append(child)

        pop = np.array(new_pop)

    return best_ind, best_score