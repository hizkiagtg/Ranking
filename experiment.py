import re
import os
import math
from bsbi import BSBIIndex
from compression import VBEPostings
from tqdm import tqdm
from collections import defaultdict
from Letor_ft import Letor
# >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP


def rbp(ranking, p=0.8):
    """ menghitung search effectiveness metric score dengan 
        Rank Biased Precision (RBP)

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score RBP
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking):
    """ menghitung search effectiveness metric score dengan 
        Discounted Cumulative Gain

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score DCG
    """
    # TODO
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] / math.log2(i + 1)
    return score


def prec(ranking, k):
    """ menghitung search effectiveness metric score dengan 
        Precision at K

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        k: int
          banyak dokumen yang dipertimbangkan atau diperoleh

        Returns
        -------
        Float
          score Prec@K
    """
    # TODO
    score = 0.
    for i in range(k):
        score += ranking[i] 
    return score / k

def ap(ranking):
    """ menghitung search effectiveness metric score dengan 
        Average Precision

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score AP
    """
    # TODO
    r = 0.
    for el in ranking:
        r += el
    if r == 0:
        return 0
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * prec(ranking, i) 
    return score / r 

# >>>>> memuat qrels


def load_qrels(qrel_file="qrels-folder/test_qrels.txt"):
    """ 
        memuat query relevance judgment (qrels) 
        dalam format dictionary of dictionary qrels[query id][document id],
        dimana hanya dokumen yang relevan (nilai 1) yang disimpan,
        sementara dokumen yang tidak relevan (nilai 0) tidak perlu disimpan,
        misal {"Q1": {500:1, 502:1}, "Q2": {150:1}}
    """
    with open(qrel_file) as file:
        content = file.readlines()

    qrels_sparse = {}

    qrels = defaultdict(lambda: defaultdict(lambda: 0)) 
    with open(qrel_file) as file:
        for line in file:
          parts = line.strip().split()
          qid = parts[0]
          did = int(parts[1])
          qrels[qid][did] = 1
    return qrels


# >>>>> EVALUASI !


def eval_retrieval(qrels, query_file="qrels-folder/test_queries.txt", k=100):
    """ 
      loop ke semua query, hitung score di setiap query,
      lalu hitung MEAN SCORE-nya.
      untuk setiap query, kembalikan top-100 documents
    """
    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    with open(query_file) as file:
       
        rbp_scores_letor = []
        dcg_scores_letor = []
        ap_scores_letor = []

        for qline in tqdm(file):
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])            
            ranking_letor = []
            # Using 100 document from BM25
            lst_did = []
            
            try:
                for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k, k1 = 1.2, b = 0.75):
                    lst_did.append(doc)
                        
                """
                Evaluasi Model
                """
                rank = letor_instance.re_ranking(query, lst_did)
                for (doc, score) in rank:
                    did = int(os.path.splitext(os.path.basename(doc))[0])
                    if (did in qrels[qid]):
                        ranking_letor.append(1)
                    else:
                        ranking_letor.append(0)
                        
                rbp_scores_letor.append(rbp(ranking_letor))
                dcg_scores_letor.append(dcg(ranking_letor))
                ap_scores_letor.append(ap(ranking_letor))

            except:
                continue

    print("Hasil evaluasi Model terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_letor) / len(rbp_scores_letor))
    print("DCG score =", sum(dcg_scores_letor) / len(dcg_scores_letor))
    print("AP score  =", sum(ap_scores_letor) / len(ap_scores_letor))


if __name__ == '__main__':
    letor_instance = Letor()
    letor_instance.main()
    qrels = load_qrels()
    eval_retrieval(qrels)
