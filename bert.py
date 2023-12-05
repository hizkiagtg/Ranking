# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:26:20 2023

@author: Hizbast
"""

import os
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class BERTReRanker:
    def __init__(self, model_name='cahya/bert-base-indonesian-522M'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tokenize(self, text):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def re_rank(self, query, docs):
        rep = self.vector_rep_bert(query)
        res_did = []
        res= []
        for did in docs:
            # Comment two lines below if error occured
            file_path = os.path.join(".", did)
            file_path = file_path.replace("\\", "/")
            with open(file_path) as f:
                res_did.append((did, f.read()))

        for did, doc_text in res_did:
            doc_score = self.score_bert(rep, doc_text)
            res.append((did, doc_score))

        sorted_did_scores = sorted(res, key=lambda tup: tup[1], reverse=True)
        return sorted_did_scores

    def vector_rep_bert(self, text):
        tokens = self.tokenize(text)
        inputs = torch.tensor([tokens]).to(self.device)

        with torch.no_grad():
            output = self.model(inputs)

        rep = output.logits.item()  
        return rep

    def score_bert(self, query_rep, doc_text):
        doc_tokens = self.tokenize(doc_text)
        inputs = torch.tensor([doc_tokens]).to(self.device)

        with torch.no_grad():
            output = self.model(inputs)

        doc_score = output.logits.item()  
        return doc_score
