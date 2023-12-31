from bsbi import BSBIIndex
from compression import VBEPostings
from letor import Letor


# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')
letor_instance = Letor("qrels-folder/train_docs.txt", "qrels-folder/train_queries.txt", "qrels-folder/train_qrels.txt")
letor_instance.main()

queries = ["Jumlah uang terbatas yang telah ditentukan sebelumnya bahwa seseorang harus membayar dari tabungan mereka sendiri",
           ]

lst_did = []

for query in queries:
    print("Query  : ", query)
    print("Results (BM25):")
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=100):
        text = f"{doc:30} {score:>.3f}"
        lst_did.append(doc)
        print(text)
        
print()
print('-'*50)
print()

# Re-Ranking with Letor

for query in queries:
    print("Query  : ", query)
    print("Results (Letor):")
    rank = letor_instance.re_ranking(query, lst_did)
    for (doc, scores) in rank:
        text = f"{doc:30} {scores:>.3f}"
        print(text)
    


    
        