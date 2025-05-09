import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SimpleDataset


def train(model, dataset, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loader_a = DataLoader(
        SimpleDataset(
            dataset["task_a"]["input_ids"],
            dataset["task_a"]["attention_mask"],
            dataset["task_a"]["labels"],
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    loader_b = DataLoader(
        SimpleDataset(
            dataset["task_b"]["input_ids"],
            dataset["task_b"]["attention_mask"],
            dataset["task_b"]["labels"],
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    model.train()

    for step, (batch_a, batch_b) in enumerate(zip(loader_a, loader_b)):
        # Task A
        logits_a = model(batch_a["input_ids"], batch_a["attention_mask"], task="A")
        loss_a = F.cross_entropy(logits_a, batch_a["labels"])

        # Task B
        logits_b = model(batch_b["input_ids"], batch_b["attention_mask"], task="B")
        loss_b = F.cross_entropy(logits_b, batch_b["labels"])

        loss = loss_a + loss_b
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_a = (logits_a.argmax(dim=1) == batch_a["labels"]).float().mean()
        acc_b = (logits_b.argmax(dim=1) == batch_b["labels"]).float().mean()

        print(
            f"Step {step:02d}: Loss A = {loss_a.item():.4f}, Acc A = {acc_a:.2f}, "
            f"Loss B = {loss_b.item():.4f}, Acc B = {acc_b:.2f}, Total = {loss.item():.4f}"
        )
