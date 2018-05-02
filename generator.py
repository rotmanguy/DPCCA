import numpy as np

def prepare_batch(dataset,split,batch_size_train):
    # split is 'train'/'val'/'test'
    dataset_split = dataset[split]
    split_size = 0
    for _ in dataset_split:
        split_size += 1
    idx_map = np.arange(split_size)
    batch_size = split_size
    if split == 'train':
        # Shuffling train
        np.random.shuffle(idx_map)
        batch_size = batch_size_train
    return [dataset_split,batch_size,idx_map]

def normalize(v):
    # Normalizing input
    norm = np.linalg.norm(v)
    if norm != 0:
        v = v / norm
    if np.isnan(v).any():
        v = np.zeros_like(v)
    return v

def centralize(view):
    # Centralizing input
    return np.subtract(view, np.mean(view, 0))

def MyGenerator(dataset,split,feats,batch_size_,train_mode):
    # Returning batches of our dataset
    [dataset_split,batch_size,idx_map] = prepare_batch(dataset,split,batch_size_)
    source = iter(idx_map)
    n = len(feats)
    while True:
        chunk = [val for _, val in zip(range(batch_size), source)]
        if not chunk:
            raise StopIteration
        next_batch = [[] for _ in range(n)]
        for c in chunk:
            feats_val = [[] for _ in range(n)]
            for feat,i in zip(feats,range(n)):
                x = dataset_split['%06d' % c][feat + '_feats'].value
                feats_val[i].append(x)
            for i in range(n):
                next_batch[i].append(feats_val[i][0])
        for i,view in enumerate(next_batch):
            next_batch[i] = normalize(centralize(np.array(view)))
        if train_mode:
            yield next_batch
        else:
            yield next_batch
            return

def Generate_Simlex_batch(languages, folder):
    batch = [[], []]
    for i, language in enumerate(languages):
        batch[i] = np.genfromtxt(
            './data_sample/simlex_data/' + folder + '/simlex_' + language + '_batch_vectors.csv',
            delimiter=",", dtype=np.float32)
    return batch