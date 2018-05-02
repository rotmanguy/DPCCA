from heapq import nlargest
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity

def knn(k, list):
    # Returning a list of top K-Nearest-Neighbours
    if k == 1:
        m = max(list)
        return [i for i, j in enumerate(list) if j == m] # In case there is more than one best score
    else:
        list_ = nlargest(k, enumerate(list), itemgetter(1))
        return [idx for idx,_ in list_]

def cosine_similarity_recall_k(view1, view2, k):
    ## Finding the top K-Nearest-Neighbours by the cosine metric ##
    if not isinstance(k, int) or k < 1:
        'k must be an integer larger than zero'
        return
    hits = 0
    res_idxs = {}
    n = len(view1)
    for idx in range(n):
        sample = [view1[idx]]
        cosine_distances = cosine_similarity(sample, view2)[0]
        result = knn(k,cosine_distances)
        if k > 1 or len(result) == 1:
            res_idxs[idx] = result
        elif idx in result and k == 1:
            res_idxs[idx] = [idx] # We would like to return only one index for k = 1
        else:
            res_idxs[idx] = [result[0]] # We would like to return only one index for k = 1
        # Checking if we got a hit
        if idx in result and len(result) <= 5:
            hits += 1
    total = idx + 1
    recall = float(hits) / (total)
    return res_idxs, recall, hits, total


## Test ##
def evaluate(F, G, cfg):
    F = F.data.numpy()
    G = G.data.numpy()
    feat_F = cfg.feats[0]
    feat_G = cfg.feats[1]
    ### Recall@k Scores ###
    k = 1
    F_res, F_recall, F_hits, F_total = cosine_similarity_recall_k(F, G, k)
    print('R@' + str(k) + ' - ' + feat_F + ' to ' + feat_G)
    print(F_hits,'/', F_total,'({:.2%})'.format(F_recall))
    G_res, G_recall, G_hits, G_total = cosine_similarity_recall_k(G, F, k)
    print('R@' + str(k) + ' - ' + feat_G + ' to ' + feat_F)
    print(G_hits,'/', G_total,'({:.2%})'.format(G_recall))
    F_recall = round(F_recall,3)
    G_recall = round(G_recall,3)
    return F_recall, G_recall