# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 23:04:50 2023

@author: Hizbast
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:07:47 2023

@author: Hizbast
"""
import math
import os
import random
import numpy as np
from gensim.models import TfidfModel, LsiModel, Word2Vec, FastText
from gensim.corpora import Dictionary
from lightgbm import sklearn
from scipy.spatial.distance import cosine
import lightgbm
from collections import defaultdict
from sklearn.model_selection import train_test_split
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score


class Letor:
    def __init__(self, train_docs_file = "qrels-folder/train_docs.txt", 
                 train_queries_file ="qrels-folder/train_queries.txt", 
                 train_qrel_file = "qrels-folder/train_qrels.txt",  
                 val_queries_file = 'qrels-folder/val_queries.txt',
                 val_qrels_file = 'qrels-folder/val_qrels.txt'):
        self.val_queries_file = val_queries_file
        self.val_qrels_file = val_qrels_file
        self.documents = self.load_documents(train_docs_file)
        self.queries = self.load_queries(train_queries_file)
        self.dataset = self.create_dataset(train_qrel_file)
        self.dictionary = Dictionary()
        self.NUM_LATENT_TOPICS = 200
        

    def load_documents(self, file_path):
        print("Loading documents...")
        documents = {}
        with open(file_path, encoding="utf8") as file:
            for line in file:
                doc_id, content = line.split(None, 1)
                documents[doc_id] = content.split()
        return documents

    def load_queries(self, file_path):
        print("Loading queries...")
        queries = {}
        with open(file_path) as file:
            for line in file:
                q_id, content = line.split(None, 1)
                queries[q_id] = content.split()
        return queries

    def create_dataset(self, qrel_file):
        print("Creating dataset...")
        NUM_NEGATIVES = 1
        q_docs_rel = defaultdict(list)

        with open(qrel_file) as file:
            for line in file:
                q_id, doc_id, rel = line.split()
                if (q_id in self.queries) and (doc_id in self.documents):
                    q_docs_rel[q_id].append((doc_id, int(rel)))

        group_qid_count = []
        dataset = []

        for q_id in q_docs_rel:
            docs_rels = q_docs_rel[q_id]
            group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)

            for doc_id, rel in docs_rels:
                dataset.append((self.queries[q_id], self.documents[doc_id], rel))

            dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))

        return dataset, group_qid_count

    def create_lsi_model(self):
        bow_corpus = [self.dictionary.doc2bow(doc, allow_update=True) for doc in self.documents.values()]
        model = LsiModel(bow_corpus, num_topics=self.NUM_LATENT_TOPICS)
        self.model = model

    def vector_rep(self, text, lsi_model):
        rep = [topic_value for (_, topic_value) in lsi_model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS

    def create_tf_idf_model(self):
        bow_corpus = [self.dictionary.doc2bow(doc, allow_update=True) for doc in self.documents.values()]
        model = TfidfModel(bow_corpus)
        self.model = model

    def vector_rep_tf_idf(self, text, tf_idf_model):
        rep = [topic_value for (_, topic_value) in tf_idf_model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS

    def create_w2v_model(self):
        all_texts = list(self.documents.values()) + list(self.queries.values())
        model = Word2Vec(all_texts, vector_size=4000, window=6, min_count=1, workers=-1)
        self.model = model
        
    def create_ft_model(self):
        all_texts = list(self.documents.values()) 
        model = FastText(all_texts, vector_size=200, window=3, min_count=2, workers=-1)
        self.model = model
       
    def vector_rep_ft(self, text, ft_model):
        rep = [ft_model.wv[word] for word in text if word in ft_model.wv]
        return np.mean(rep, axis=0).tolist() if len(rep) > 0 else [0.] * self.NUM_LATENT_TOPICS
   
    def vector_rep_w2v(self, text, w2v_model):
        rep = [w2v_model.wv[word] for word in text if word in w2v_model.wv]
        return np.mean(rep, axis=0).tolist() if len(rep) > 0 else [0.] * self.NUM_LATENT_TOPICS

    def features(self, query, doc, model):
        v_q = self.vector_rep_ft(query, model)
        v_d = self.vector_rep_ft(doc, model)

        q = set(query)
        d = set(doc)

        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)

        result = v_q + v_d + [jaccard] \
                 + [cosine_dist]
        return result

    def prepare_data(self):
        X = []
        Y = []

        for (query, doc, rel) in self.dataset[0]:
            X.append(self.features(query, doc, self.model))
            Y.append(rel)

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def prepare_eval(self):

        self.val_queries = self.load_queries(self.val_queries_file)
        self.val_dataset = self.create_dataset(self.val_qrels_file)

        # Create feature matrix and labels for validation set
        lst = []

        for (query, doc, rel) in self.val_dataset[0]:
            x = self.features(query, doc, self.model)
            y = rel
            lst.append((x, y))

        return self.val_dataset[1], np.array(lst)
    
    def train_lambda_mart(self, X, Y):
        print("Training LambdaMART...")
        ranker = lightgbm.LGBMRanker(
            objective="lambdarank",
            boosting_type="gbdt",
            n_estimators=128,
            importance_type="split",
            metric="ndcg",
            num_leaves=40,
            learning_rate=0.05960767981428276,
            max_depth=10
        )

        # Menggunakan validation set
        eval_group, evals = self.prepare_eval()
        ranker.fit(X, Y, group=self.dataset[1], eval_set=evals, eval_group=eval_group)

        return ranker
    
    def make_predictions(self, X_unseen):
        return self.ranker.predict(X_unseen)

    def re_ranking(self, query, docs):
        res = []
        for did in docs:
            # Comment two lines below if error occured
            file_path = os.path.join(".", did)
            file_path = file_path.replace("\\", "/")
            with open(file_path) as f:
                res.append((did, f.read()))
        X_unseen = [self.features(query.split(), doc.split(), self.model) for (_, doc) in res]
        X_unseen = np.array(X_unseen)

        scores = self.make_predictions(X_unseen)

        did_scores = [x for x in zip([did for (did, _) in res], scores)]
        sorted_did_scores = sorted(did_scores, key=lambda tup: tup[1], reverse=True)

        return sorted_did_scores
    
    def main(self):
        self.create_ft_model()
        X, Y = self.prepare_data()
        self.ranker = self.train_lambda_mart(X, Y)
        

    def objective(self, trial):
        x_train, y_train = self.prepare_data()
        group_qids_val, t = self.prepare_eval()
        group_qids_train=self.dataset[1]
        
        ranking_param_grid = {
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "importance_type": trial.suggest_categorical("importance_type", ["gain", "split"]),
            "metric": trial.suggest_categorical("metric", ["ndcg"]),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
            "max_depth": trial.suggest_int("max_depth", -1, 20),
        }
        
        gbm = lightgbm.LGBMRanker(n_jobs=24, **ranking_param_grid)
        gbm.fit(x_train, y_train, group=group_qids_train) 
        print(gbm.best_score_['valid_0'])
        return gbm.best_score_['valid_0'].get('metric')


    def optimize_hyperparameters(self, n_trials=10):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)

        print('Number of finished trials:', len(study.trials))
        print('Best trial:')
        print(study.best_trial.params)

        X, Y = self.prepare_data()
        self.ranker = self.train_lambda  