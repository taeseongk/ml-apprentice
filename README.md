# Multi-Task Sentence Transformer

This project implements a multi-task learning framework using a shared BERT encoder to perform:
- **Task A**: Sentence classification (e.g., topic classification using AG News)
- **Task B**: Sentiment analysis (e.g., binary classification using IMDB)

The model uses PyTorch and Hugging Face Transformers, and it is structured to be easily trainable and testable using Docker.

---

## Project Structure

├── main.py # Entry point for training and evaluation
├── task1.py # Basic sentence transformer encoder
├── task2.py # Multi-task learning architecture
├── dataset.py # Dataset loading and preprocessing
├── train.py # Training loop for MTL
├── validate.py # Validation loop for MTL
├── requirements.txt # Python dependencies
├── Dockerfile # Container setup
└── README.md # Project instructions


### Build the Docker image:

```bash
docker build -t ml-apprentice-app .

dokcer run --rm ml-apprentice-app

### Run it manually:
fd
