# Task 1: Sentence Transformer Implementation

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SentenceTransformer(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, sentences):
        enc = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        )
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = self.mean_pooling(outputs, attention_mask)
        return sentence_embeddings


model = SentenceTransformer()
sentences = ["I love machine learning.", "Transformers are powerful."]
embeddings = model.encode(sentences)

print("Embeddings:\n", embeddings)

"""
I have decided to use BERT as my pre-trained transformer model, as it is a strong general-purpose language model,
and its token embeddings are contextualized with respect to the full sentence. Although smaller models such as MiniLM
offer faster inference and optimized embeddings, I chose bert-base-uncased to demonstrate full control over the sentence
embedding pipeline.

I have chosen mean pooling as my pooling strategy, as it is simple and effective. It creates sentence embeddings
that are robust for general-purpose similarity and classification tasks. I have also decided on fine-tuning in later stages
when we attach task-specific heads.
"""
