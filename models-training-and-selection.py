# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments, GPT2Tokenizer, GPT2Model, AdamW, TrainerCallback
from torch.utils.data import DataLoader, Dataset
import joblib
import os
import optuna


# Define the save path
save_path = r"C:\Users\aanas\Desktop\IronWork\GitHub\Music-recommendation-project\models"
os.makedirs(save_path, exist_ok=True)

# Paths to data 
train_data_path = r"C:\Users\aanas\Desktop\IronWork\GitHub\Music-recommendation-project\datasets\lyrics_embeddings_reduced.csv"
test_data_path = r"C:\Users\aanas\Desktop\IronWork\GitHub\Music-recommendation-project\datasets\lyrics_embeddings.csv"

# Load the data
reduced_embeddings_df = pd.read_csv(train_data_path)
embeddings_df = pd.read_csv(test_data_path)

# Extract features and labels
X = np.array(reduced_embeddings_df['reduced_embedding'].apply(eval).tolist())
y = reduced_embeddings_df['mood']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# For BERT and GPT models, prepare text and labels
train_texts = embeddings_df['lyrics'].tolist()
train_labels = embeddings_df['mood'].tolist()

# Encode labels if necessary
label_to_id = {label: idx for idx, label in enumerate(set(train_labels))}
id_to_label = {idx: label for label, idx in label_to_id.items()}
train_labels = [label_to_id[label] for label in train_labels]

# Define parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16  # Reduce batch size for better memory management

# %%
# 1. Support Vector Machine (SVM)
print("Training SVM...")
svm = SVC()
svm_parameters = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]}
svm_grid_search = GridSearchCV(svm, svm_parameters, cv=3, scoring='accuracy')
svm_grid_search.fit(X_train, y_train)
best_svm = svm_grid_search.best_estimator_

# Save the trained SVM model
svm_model_path = os.path.join(save_path, 'svm_model.joblib')
joblib.dump(best_svm, svm_model_path)
print(f"SVM model saved to {svm_model_path}")

# Evaluate SVM
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Evaluate SVM and save predictions
svm_predictions = best_svm.predict(X_test)

# Calculate accuracy for SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy}")

# Generate classification report for SVM
svm_classification_report = classification_report(y_test, svm_predictions)
print("SVM Classification Report:")
print(svm_classification_report)

# Save accuracy and classification report to a text file
svm_report_path = os.path.join(save_path, 'svm_report.txt')
with open(svm_report_path, 'w') as f:
    f.write(f"SVM Accuracy: {svm_accuracy}\n\n")
    f.write("SVM Classification Report:\n")
    f.write(svm_classification_report)
print(f"SVM report saved to {svm_report_path}")

# %%
# Custom callback for Optuna pruning
class TransformersPruningCallback(TrainerCallback):
    def __init__(self, trial, monitor):
        self.trial = trial
        self.monitor = monitor

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            metrics = {}
        value = metrics.get(self.monitor)
        if value is None:
            return
        self.trial.report(value, state.global_step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# %%
# 2. BERT Model with Hyperparameter Tuning
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BertLyricsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):  # Reduced max length
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

train_dataset = BertLyricsDataset(train_texts, train_labels, tokenizer)

def model_init():
    return BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(y)))

def objective(trial):
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=trial.suggest_int('num_train_epochs', 1, 3),
        per_device_train_batch_size=trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32]),
        per_device_eval_batch_size=trial.suggest_categorical('per_device_eval_batch_size', [8, 16, 32]),
        warmup_steps=trial.suggest_int('warmup_steps', 100, 500),
        weight_decay=trial.suggest_float('weight_decay', 0.01, 0.1),
        logging_dir='./logs',
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 5e-5),
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # Using train_dataset for simplicity; replace with actual validation dataset
        callbacks=[TransformersPruningCallback(trial, monitor="eval_loss")]
    )

    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results['eval_loss']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

best_trial = study.best_trial
print(f"Best trial parameters: {best_trial.params}")

# Use best hyperparameters to train the final model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=best_trial.params['num_train_epochs'],
    per_device_train_batch_size=best_trial.params['per_device_train_batch_size'],
    per_device_eval_batch_size=best_trial.params['per_device_eval_batch_size'],
    warmup_steps=best_trial.params['warmup_steps'],
    weight_decay=best_trial.params['weight_decay'],
    logging_dir='./logs',
    learning_rate=best_trial.params['learning_rate'],
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model_init(),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset  # Using train_dataset for simplicity; replace with actual validation dataset
)

trainer.train()

# Save the trained BERT model
bert_model_path = os.path.join(save_path, 'bert_model')
trainer.save_model(bert_model_path)
tokenizer.save_pretrained(bert_model_path)
print(f"BERT model saved to {bert_model_path}")

# Evaluate BERT on test data
test_texts = embeddings_df['lyrics'].tolist()
test_labels = [label_to_id[label] for label in embeddings_df['mood'].tolist()]
test_dataset = BertLyricsDataset(test_texts, test_labels, tokenizer)

predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)

# Calculate accuracy for BERT
bert_accuracy = accuracy_score(test_labels, pred_labels)
print(f"BERT Accuracy: {bert_accuracy}")

# Generate classification report for BERT
bert_classification_report = classification_report(test_labels, pred_labels)
print("BERT Classification Report:")
print(bert_classification_report)

# Save accuracy and classification report to a text file
bert_report_path = os.path.join(save_path, 'bert_report.txt')
with open(bert_report_path, 'w') as f:
    f.write(f"BERT Accuracy: {bert_accuracy}\n\n")
    f.write("BERT Classification Report:\n")
    f.write(bert_classification_report)
print(f"BERT report saved to {bert_report_path}")

# %%
#3. GPT Model
num_classes = len(set(train_labels))  # Calculate the number of classes

# Custom Dataset class for GPT
class GPTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize the GPT2 tokenizer globally
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters for the trial
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    num_epochs = trial.suggest_int('num_epochs', 1, 5)
    max_length = trial.suggest_int('max_length', 64, 256, step=64)

    # Prepare the dataset with the suggested batch size and max length
    gpt_train_dataset = GPTDataset(train_texts, train_labels, gpt_tokenizer, max_length=max_length)
    gpt_train_loader = DataLoader(gpt_train_dataset, batch_size=batch_size, shuffle=True)

    gpt_model = GPT2Model.from_pretrained('gpt2')
    gpt_model.resize_token_embeddings(len(gpt_tokenizer))
    gpt_model.to(device)

    classifier_head = nn.Linear(gpt_model.config.hidden_size, num_classes)
    classifier_head.to(device)

    optimizer = AdamW(list(gpt_model.parameters()) + list(classifier_head.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    gpt_model.train()
    for epoch in range(num_epochs):
        for batch in gpt_train_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = gpt_model(inputs, attention_mask=attention_mask)[0]
            logits = classifier_head(outputs[:, -1, :])
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate the model on the validation set
    gpt_model.eval()
    with torch.no_grad():
        gpt_test_dataset = GPTDataset(train_texts, train_labels, gpt_tokenizer, max_length=max_length)
        gpt_test_loader = DataLoader(gpt_test_dataset, batch_size=batch_size)
        all_preds = []
        all_labels = []
        for batch in gpt_test_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = gpt_model(inputs, attention_mask=attention_mask)[0]
            logits = classifier_head(outputs[:, -1, :])
            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Get the best hyperparameters
best_params = study.best_trial.params
print("Best hyperparameters found:")
print(best_params)

# Use the best hyperparameters to train the final model
best_learning_rate = best_params['learning_rate']
best_batch_size = best_params['batch_size']
best_num_epochs = best_params['num_epochs']
best_max_length = best_params['max_length']

# Prepare the dataset with the best hyperparameters
gpt_train_dataset = GPTDataset(train_texts, train_labels, gpt_tokenizer, max_length=best_max_length)
gpt_train_loader = DataLoader(gpt_train_dataset, batch_size=best_batch_size, shuffle=True)

gpt_model = GPT2Model.from_pretrained('gpt2')
gpt_model.resize_token_embeddings(len(gpt_tokenizer))
gpt_model.to(device)

classifier_head = nn.Linear(gpt_model.config.hidden_size, num_classes)
classifier_head.to(device)

optimizer = AdamW(list(gpt_model.parameters()) + list(classifier_head.parameters()), lr=best_learning_rate)
criterion = nn.CrossEntropyLoss()

# Train the final model
gpt_model.train()
for epoch in range(best_num_epochs):
    for batch in gpt_train_loader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = gpt_model(inputs, attention_mask=attention_mask)[0]
        logits = classifier_head(outputs[:, -1, :])
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{best_num_epochs}], Loss: {loss.item():.4f}")

# Save the trained GPT model
gpt_model_path = os.path.join(save_path, 'gpt_model')
os.makedirs(gpt_model_path, exist_ok=True)
gpt_model.save_pretrained(gpt_model_path)
gpt_tokenizer.save_pretrained(gpt_model_path)
torch.save(classifier_head.state_dict(), os.path.join(gpt_model_path, 'classifier_head.pth'))
print(f"GPT model saved to {gpt_model_path}")

# Evaluate GPT Model on test data
test_texts = embeddings_df['lyrics'].tolist()
test_labels = [label_to_id[label] for label in embeddings_df['mood'].tolist()]
gpt_test_dataset = GPTDataset(test_texts, test_labels, gpt_tokenizer, max_length=best_max_length)
gpt_test_loader = DataLoader(gpt_test_dataset, batch_size=best_batch_size)

gpt_model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in gpt_test_loader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = gpt_model(inputs, attention_mask=attention_mask)[0]
        logits = classifier_head(outputs[:, -1, :])
        _, preds = torch.max(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy for GPT
gpt_accuracy = accuracy_score(all_labels, all_preds)
print(f"GPT Accuracy: {gpt_accuracy}")

# Generate classification report for GPT
gpt_classification_report = classification_report(all_labels, all_preds)
print("GPT Classification Report:")
print(gpt_classification_report)

# Save accuracy and classification report to a text file
gpt_report_path = os.path.join(save_path, 'gpt_report.txt')
with open(gpt_report_path, 'w') as f:
    f.write(f"GPT Accuracy: {gpt_accuracy}\n\n")
    f.write("GPT Classification Report:\n")
    f.write(gpt_classification_report)
print(f"GPT report saved to {gpt_report_path}")

# %%
# Get the best hyperparameters
best_params = study.best_trial.params
print("Best hyperparameters found:")
print(best_params)

# Use the best hyperparameters to train the final model
best_learning_rate = best_params['learning_rate']
best_batch_size = best_params['batch_size']
best_num_epochs = best_params['num_epochs']
best_max_length = best_params['max_length']

# Prepare the dataset with the best hyperparameters
gpt_train_dataset = GPTDataset(train_texts, train_labels, gpt_tokenizer, max_length=best_max_length)
gpt_train_loader = DataLoader(gpt_train_dataset, batch_size=best_batch_size, shuffle=True)

gpt_model = GPT2Model.from_pretrained('gpt2')
gpt_model.resize_token_embeddings(len(gpt_tokenizer))
gpt_model.to(device)

classifier_head = nn.Linear(gpt_model.config.hidden_size, num_classes)
classifier_head.to(device)

optimizer = AdamW(list(gpt_model.parameters()) + list(classifier_head.parameters()), lr=best_learning_rate)
criterion = nn.CrossEntropyLoss()

# Train the final model
gpt_model.train()
for epoch in range(best_num_epochs):
    for batch in gpt_train_loader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = gpt_model(inputs, attention_mask=attention_mask)[0]
        logits = classifier_head(outputs[:, -1, :])
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{best_num_epochs}], Loss: {loss.item():.4f}")

# Save the trained GPT model
gpt_model_path = os.path.join(save_path, 'gpt_model')
os.makedirs(gpt_model_path, exist_ok=True)
gpt_model.save_pretrained(gpt_model_path)
gpt_tokenizer.save_pretrained(gpt_model_path)
torch.save(classifier_head.state_dict(), os.path.join(gpt_model_path, 'classifier_head.pth'))
print(f"GPT model saved to {gpt_model_path}")

# Evaluate GPT Model on test data
test_texts = embeddings_df['lyrics'].tolist()
test_labels = [label_to_id[label] for label in embeddings_df['mood'].tolist()]
gpt_test_dataset = GPTDataset(test_texts, test_labels, gpt_tokenizer, max_length=best_max_length)
gpt_test_loader = DataLoader(gpt_test_dataset, batch_size=best_batch_size)

gpt_model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in gpt_test_loader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = gpt_model(inputs, attention_mask=attention_mask)[0]
        logits = classifier_head(outputs[:, -1, :])
        _, preds = torch.max(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy for GPT
gpt_accuracy = accuracy_score(all_labels, all_preds)
print(f"GPT Accuracy: {gpt_accuracy}")

# Generate classification report for GPT
gpt_classification_report = classification_report(all_labels, all_preds)
print("GPT Classification Report:")
print(gpt_classification_report)

# Save accuracy and classification report to a text file
gpt_report_path = os.path.join(save_path, 'gpt_report.txt')
with open(gpt_report_path, 'w') as f:
    f.write(f"GPT Accuracy: {gpt_accuracy}\n\n")
    f.write("GPT Classification Report:\n")
    f.write(gpt_classification_report)
print(f"GPT report saved to {gpt_report_path}")

# %%
# Summary of model performances
print("Model Performance Summary:")
print(f"SVM Accuracy: {svm_accuracy}")
print(f"BERT Accuracy: {bert_accuracy}")
print(f"GPT Accuracy: {gpt_accuracy}")

# Save the best model
best_model_path = os.path.join(save_path, 'best_model')
if bert_accuracy > max(svm_accuracy, gpt_accuracy):
    model.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"Best model (BERT) saved to {best_model_path}")
elif svm_accuracy > gpt_accuracy:
    joblib.dump(best_svm, os.path.join(best_model_path, 'svm_model.joblib'))
    print(f"Best model (SVM) saved to {best_model_path}")
else:
    torch.save(gpt_model.state_dict(), os.path.join(best_model_path, 'gpt_model.pth'))
    print(f"Best model (GPT) saved to {best_model_path}")


