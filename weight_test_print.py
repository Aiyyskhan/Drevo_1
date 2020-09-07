import numpy as np

def loading(path):
    with open(path, 'rb') as f:
        len_weights = np.load(f, allow_pickle=True)
        
        parameters = np.load(f, allow_pickle=True)
        indices = np.load(f, allow_pickle=True)
        weights = []
        
        for _ in range(len_weights):
            weights.append(np.load(f, allow_pickle=True))
    return parameters, indices, weights

if __name__ == "__main__":
    path = "data/level_5_and_0_20200901/individ_5.3.30.gen2_4.npy"

    p, i, w = loading(path)

    print(p, "\n")
    print(i, "\n")
    print(w, "\n")