from datasets import load_dataset
import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def load(dataset_a, dataset_b, split):
    return load_dataset(dataset_a, split=split), load_dataset(dataset_b, split=split)


def load_combined_dataset(model, dataset_a, dataset_b, max_samples=1000):
    tokenizer = model.tokenizer

    # Task A: AG News -> 4-class topic classification
    dataset_a = dataset_a.shuffle()
    dataset_a = dataset_a.select(range(max_samples))
    dataset_a_texts = dataset_a["text"]
    dataset_a_labels = torch.tensor(dataset_a["label"])

    task_a_enc = tokenizer(
        dataset_a_texts, return_tensors="pt", padding=True, truncation=True
    )
    task_a_input_ids, task_a_attention_mask = (
        task_a_enc["input_ids"],
        task_a_enc["attention_mask"],
    )

    # Task B: IMDB -> binary sentiment classification
    dataset_b = dataset_b.shuffle()
    dataset_b = dataset_b.select(range(max_samples))
    dataset_b_texts = dataset_b["text"]
    dataset_b_labels = torch.tensor(dataset_b["label"])

    task_b_enc = tokenizer(
        dataset_b_texts, return_tensors="pt", padding=True, truncation=True
    )
    task_b_input_ids, task_b_attention_mask = (
        task_b_enc["input_ids"],
        task_b_enc["attention_mask"],
    )

    return {
        "task_a": {
            "input_ids": task_a_input_ids,
            "attention_mask": task_a_attention_mask,
            "labels": dataset_a_labels,
        },
        "task_b": {
            "input_ids": task_b_input_ids,
            "attention_mask": task_b_attention_mask,
            "labels": dataset_b_labels,
        },
    }
