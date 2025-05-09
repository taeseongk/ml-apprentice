import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SimpleDataset


def validate(model, dataset, batch_size):
    model.eval()
    loss_a, corr_a, samples_a = 0, 0, 0
    loss_b, corr_b, samples_b = 0, 0, 0

    loader_a = DataLoader(
        SimpleDataset(
            dataset["task_a"]["input_ids"],
            dataset["task_a"]["attention_mask"],
            dataset["task_a"]["labels"],
        ),
        batch_size=batch_size,
    )
    loader_b = DataLoader(
        SimpleDataset(
            dataset["task_b"]["input_ids"],
            dataset["task_b"]["attention_mask"],
            dataset["task_b"]["labels"],
        ),
        batch_size=batch_size,
    )
    with torch.no_grad():
        for batch in loader_a:
            logits = model(batch["input_ids"], batch["attention_mask"], task="A")
            loss = F.cross_entropy(logits, batch["labels"])

            preds = torch.argmax(logits, dim=1)
            correct = (preds == batch["labels"]).sum().item()

            loss_a += loss.item() * len(batch["labels"])
            corr_a += correct
            samples_a += len(batch["labels"])

        for batch in loader_b:
            logits = model(batch["input_ids"], batch["attention_mask"], task="B")
            loss = F.cross_entropy(logits, batch["labels"])

            preds = torch.argmax(logits, dim=1)
            correct = (preds == batch["labels"]).sum().item()

            loss_b += loss.item() * len(batch["labels"])
            corr_b += correct
            samples_b += len(batch["labels"])

    print(
        f"[Validation] Task A - Loss: {loss_a / samples_a:.4f}, Acc: {corr_a / samples_a:.2f}"
    )
    print(
        f"[Validation] Task B - Loss: {loss_b / samples_b:.4f}, Acc: {corr_b / samples_b:.2f}"
    )
