from task2 import SentenceTransformer
from dataset import load, load_combined_dataset
from train import train
from validate import validate


def main():
    model = SentenceTransformer()
    print("HELLO")

    dataset_a, dataset_b = load("ag_news", "imdb", split="train")
    dataset = load_combined_dataset(model, dataset_a, dataset_b, 500)

    val_dataset_a, val_dataset_b = load("ag_news", "imdb", split="test")
    val_dataset = load_combined_dataset(model, val_dataset_a, val_dataset_b, 100)

    batch_size = 4
    train(model, dataset, batch_size)
    validate(model, val_dataset, batch_size)


if __name__ == "__main__":
    main()
