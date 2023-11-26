import os
import pickle
import contextlib
import heapq
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, merge_and_sort_posts_and_tfs
from compression import VBEPostings
from tqdm import tqdm

from mpstemmer import MPStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from operator import itemgetter
import regex as re

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.dl = 0 

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)
        with open(os.path.join(self.output_dir, 'dl'), 'wb') as f:
            pickle.dump([self.dl], f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'dl'), 'rb') as f:
            self.dl = pickle.load(f)


    def pre_processing_text(self, content):
        """
        Melakukan preprocessing pada text, yakni stemming dan removing stopwords
        """
        # https://github.com/ariaghora/mpstemmer/tree/master/mpstemmer

        stemmer = MPStemmer()
        stemmed = stemmer.stem(content)
        remover = StopWordRemoverFactory().create_stop_word_remover()
        return remover.remove(stemmed)

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk stemming bahasa Indonesia, seperti
        MpStemmer: https://github.com/ariaghora/mpstemmer 
        Jangan gunakan PySastrawi untuk stemming karena kode yang tidak efisien dan lambat.

        JANGAN LUPA BUANG STOPWORDS! Kalian dapat menggunakan PySastrawi 
        untuk menghapus stopword atau menggunakan sumber lain seperti:
        - Satya (https://github.com/datascienceid/stopwords-bahasa-indonesia)
        - Tala (https://github.com/masdevid/ID-Stopwords)

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        # TODO
        # Stop Words menggunakan PySastrawi
        stop_factory = StopWordRemoverFactory()
        stop_words_list = stop_factory.get_stop_words()
        stop_words_set  = set(stop_words_list)
        
        # Data path block
        data_path_block = os.path.join(self.data_dir, block_path)
        entries = sorted(os.listdir(data_path_block))
        
        # Result list of tuple
        result = []
        
        # Regex for tokenization
        tokenizer_pattern = r'\w+'

        # For stemming and lemmatization
        stemmer = MPStemmer()
        for entry in entries:
            doc_id = os.path.join(data_path_block, entry)
            doc_id_ = self.doc_id_map[doc_id]
            with open(doc_id, 'rt', encoding='utf8') as file:
                file = file.read().strip()
                tokens =  re.findall(tokenizer_pattern, file.lower())
                for token in tokens:
                    if token != '' or token != None:
                        token = stemmer.stem(token)
                        if token not in stop_words_set:
                            token_ = self.term_id_map[token] 
                            result.append((token_, doc_id_))
        return result

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
            
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
            term_dict[term_id].add(doc_id)
        for term_id in sorted(term_dict.keys()):
            index.append(term_id, sorted(list(term_dict[term_id])))
        """
        # TODO
        term_dict = {}

        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = dict()
            if doc_id not in term_dict[term_id]:
                term_dict[term_id][doc_id] = 1
            else:
                term_dict[term_id][doc_id] += 1

        for term_id in sorted(term_dict.keys()):
            doc_ids = sorted(list(term_dict[term_id].keys()))
            tf_list = []
            for tf in doc_ids:
                val = term_dict[term_id][tf]
                self.dl += val 
                tf_list.append(val)
            index.append(term_id, doc_ids, tf_list)

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        # Load data doc and term
        self.load()
        
        # Stopword
        stop_factory = StopWordRemoverFactory()
        stop_words_list = stop_factory.get_stop_words()
        stop_words_set  = set(stop_words_list)

        # Result of list documentS
        result = []
        
        # List of query
        lst_query = []
        
        # Regex for tokenization
        tokenizer_pattern = r'\w+'
        
        # For stemming and lemmatization
        stemmer = MPStemmer()
        
        
        tokens =  re.findall(tokenizer_pattern, query)
        for token in tokens:
            token = stemmer.stem(token).lower()
            if token != '' and token != None and token not in stop_words_set:
                lst_query.append(token)


        
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as index:
            N = len(index.doc_length)
            scores = [0.0]*N
            for elemen in lst_query:
                check = len(self.term_id_map)
                id_term = self.term_id_map[elemen]
                if id_term >= check:
                    continue
                else:
                    posting, tf_list = index.get_postings_list(id_term)
                    dft = index.postings_dict[id_term][1]
                    idf = math.log10(N/dft)
                    for i in range(len(posting)):
                        tf = tf_list[i]
                        if tf > 0:
                            scores[posting[i]] += (math.log10(tf) + 1)*idf
                        else:
                            pass
        heap = []
        heapq.heapify(heap)
        for i in range(N):
            heapq.heappush(heap, (scores[i]*-1, i))
        for i in range(k):
            res = heapq.heappop(heap)
            result.append((res[0]*-1, self.doc_id_map[res[1]]))
        return result

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25 dan framework TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        """
        # TODO
        # Load data doc and term
        self.load()

        # Stopword
        stop_factory = StopWordRemoverFactory()
        stop_words_list = stop_factory.get_stop_words()
        stop_words_set  = set(stop_words_list)

        # Result of list documentS
        result = []
        
        # List of query
        lst_query = []
        
        # Regex for tokenization
        tokenizer_pattern = r'\w+'
        
        # For stemming and lemmatization
        stemmer = MPStemmer()
        
        tokens =  re.findall(tokenizer_pattern, query)
        for token in tokens:
            token = stemmer.stem(token).lower()
            if token != '' and token != None and token not in stop_words_set:
                lst_query.append(token)


        
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as index:
            N = len(index.doc_length)
            scores = [0.0]*N
            avdl = self.dl[0]/N
            for elemen in lst_query:
                check = len(self.term_id_map)
                id_term = self.term_id_map[elemen]
                if id_term >= check:
                    continue
                else:
                    posting, tf_list = index.get_postings_list(id_term)
                    dft = index.postings_dict[id_term][1]
                    idf = math.log10(N/dft)
                    for i in range(len(posting)):
                        dl_ = index.doc_length[posting[i]]
                        norm = dl_/(avdl)
                        tf = tf_list[i]
                        num = (k1 + 1)*tf
                        denum = k1*((1-b)+b*norm) + tf
                        total = num/denum*idf
                        scores[posting[i]] += total
                        
        heap = []
        heapq.heapify(heap)
        for i in range(N):
            heapq.heappush(heap, (scores[i]*-1, i))
        for i in range(k):
            res = heapq.heappop(heap)
            result.append((res[0]*-1, self.doc_id_map[res[1]]))
        return result

    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.do_indexing()  # memulai indexing!

