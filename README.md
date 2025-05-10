# Multi-Task Sentence Transformer

This project implements a multi-task learning framework using a shared BERT encoder to perform:
- **Task A**: Sentence classification (e.g., topic classification using AG News)
- **Task B**: Sentiment analysis (e.g., binary classification using IMDB)

The model uses PyTorch and Hugging Face Transformers, and it is structured to be easily trainable and testable using Docker.

---

## Project Structure

```
├── main.py # Entry point for training and evaluation
├── task1.py # Basic sentence transformer encoder
├── task2.py # Multi-task learning architecture
├── dataset.py # Dataset loading and preprocessing
├── train.py # Training loop for MTL
├── validate.py # Validation loop for MTL
├── requirements.txt # Python dependencies
├── Dockerfile # Container setup
└── README.md # Project instructions
```

---

### Build the Docker image:

```bash
docker build -t ml-apprentice-app .

docker run --rm ml-apprentice-app
```

### Run it manually:

```bash
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python main.py
```

### Assessment answers

Below, I write out the answers to the questions of the assessment

#### Task 1:
To see the samples sentences and the obtained embeddings:
```bash
python task1.py
```
The architecturual decisions are explained in the file as comments.

#### Task 2:
The architectural decisions are explained in the file as comments.

#### Task 3:
1. If the entire network should be frozen.
- The implications are that no weights in the model are updated during training. 
It is useful when there is very limited labeled data and we want to avoid overfitting. However, task-specific 
heads cannot adapt to specific data and likely underperforms on domain-specific tasks.
2. If only the transformer backbone should be frozen.
- The implications are that the pre-trained BERT encoder is kept fixed and only the task-specific heads are trainable.
It's beneficial as it preserves the general language understanding captured by BERT and reduces computation and memory
requirements.
3. If only one of the task-specific heads (either for Task A or Task B) should be frozen.
- The implications are that one head (let's say for task A) is frozen, and the other (task B) is trainable. The shared
backbone still learns from the trainable head. This allows one to focus training on one task while preserving performance
on the frozen task. This is especially good when you want to add a new task without degrading performance on an old one.

Given a transfer learning scenario, let's say that we want to adapt to a new domain-specific task, such as medical document
classification.
1. The choice of a pre-trained model.
- We can use a domain-relevant model for biomedical text, or stick with bert-base-uncased for general data.
2. The layers you would freeze/unfreeze
- We can freeze early layers of the transformer and unfreeze later layers and task head.
3. The rationale behind these choices.
- By freezing the lower layers, we prevent overfitting and saves computation. Higher layers and the classification
head capture task-specific features - unfreezing them allows adaptation to new data. This balances generalization and adaptation,
which is the core of transfer learning.

#### Task 4:
The architectural decisions are explained in the files train.py and dataset.py.

This completes the assessment. Thanks for reviewing!
