# Task 2: Multi-Task Learning Expansion

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SentenceTransformer(nn.Module):
    def __init__(
        self, model_name="bert-base-uncased", num_classes_a=4, num_classes_b=2
    ):
        """
        I retained the pretrained BERT model as a shared encoder. This allows both tasks to benefit from a common set
        of contextualized sentence embeddings, improving generalization and efficiency.
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        """
        I introduced two separate fully connected heads on top of the shared encoder:
            head_a: for Task A (4-class sentence classification)
            head_b: for Task B (binary sentiment classification)
        Each head is composed of:
            A linear transformation
            ReLU activation
            Dropout of regularization
            A final linear layer projecting to the number of task-specific output classes
        """
        self.head_a = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes_a),
        )  # Task A
        self.head_b = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes_b),
        )  # Task B

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    """
    The forward method accepts a task argument to control which head is used. This enables the same model
    instance to dynamically switch between tasks during training or inference.
    """

    def forward(self, input_ids, attention_mask, task="A"):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(output, attention_mask)
        return self.head_a(pooled) if task == "A" else self.head_b(pooled)

    def encode(self, sentences):
        enc = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        )
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.mean_pooling(outputs, attention_mask)


"""
These changes introduce parameter sharing via a common encoder while maintaining task-specific specialization via
independent heads. This approach balances efficiency and task differentiation, enabling the model to generalize well across
tasks with minimal added complexity.
"""
