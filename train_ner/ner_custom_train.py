import spacy
from spacy.training import Example
from spacy.pipeline.textcat import Config
import random
from train_ner.training_data_ner import TRAIN_DATA

# Load the pre-trained model
nlp = spacy.load("en_core_web_sm")

# Get the NER component
ner = nlp.get_pipe("ner")

# Add new entity labels if not already present
new_labels = {"SECTOR"} 
for label in new_labels:
    if label not in ner.labels:
        ner.add_label(label)

# Initialize the optimizer
optimizer = nlp.resume_training()

# Training loop
for epoch in range(10):  # Number of epochs
    
    losses = {}
    # Shuffle training data
    random.shuffle(TRAIN_DATA)
    
    # Iterate over the training data
    for text, annotations in TRAIN_DATA:
        # Create Example
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        # Update the model
        nlp.update([example], drop=0.5, losses=losses)
    
    print(f"Epoch {epoch}: {losses}")

# Save the trained model
nlp.to_disk("ner_spacy_trained")
