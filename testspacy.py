import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random

# Augmented Training Data
TRAIN_DATA = [
    ("Who is Talha Tayyab?", {"entities": [(7, 19, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
    ("Agra is famous for Tajmahal.", {"entities": [(0, 4, "LOC"), (22, 30, "LOC")]}),
    ("The CEO of Facebook will visit India to meet Murari Mahaseth.", {"entities": [(12, 20, "ORG"), (32, 37, "GPE"), (42, 57, "PERSON")]}),
    ("Tajmahal is one of the seven wonders of the world.", {"entities": [(0, 8, "LOC")]}),
    ("Barack Obama served as the president of the USA.", {"entities": [(0, 12, "PERSON"), (38, 41, "GPE")]}),
    ("Jeff Bezos founded Amazon.", {"entities": [(0, 10, "PERSON"), (19, 25, "ORG")]}),
    ("Elon Musk is the CEO of Tesla.", {"entities": [(0, 9, "PERSON"), (23, 28, "ORG")]}),
    ("The Eiffel Tower is located in Paris.", {"entities": [(4, 16, "LOC"), (30, 35, "LOC")]}),
    ("Google is based in California.", {"entities": [(0, 6, "ORG"), (17, 27, "LOC")]}),
]

# Split TRAIN_DATA into Train and Test sets
random.shuffle(TRAIN_DATA)
split = int(0.8 * len(TRAIN_DATA))
train_data = TRAIN_DATA[:split]
test_data = TRAIN_DATA[split:]

# Function to train the NER model
def train_ner_model(train_data, iterations=20):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner", last=True)

    # Add entity labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable other components during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for i in range(iterations):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    examples.append(Example.from_dict(nlp.make_doc(text), annotations))
                nlp.update(examples, drop=0.3, losses=losses)
            print(f"Iteration {i+1} Loss: {losses['ner']:.4f}")

    return nlp

# Train the model
custom_ner_model = train_ner_model(train_data, iterations=15)

# Function to evaluate the model
def evaluate_model(ner_model, examples):
    tp, fp, fn = 0, 0, 0
    for text, annotations in examples:
        doc = ner_model(text)
        gold_entities = annotations["entities"]
        predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        for ent in predicted_entities:
            if ent in gold_entities:
                tp += 1
            else:
                fp += 1

        for ent in gold_entities:
            if ent not in predicted_entities:
                fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score

# Evaluate on the test set
precision, recall, f1_score = evaluate_model(custom_ner_model, test_data)

# Print the results
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1_score:.3f}")
