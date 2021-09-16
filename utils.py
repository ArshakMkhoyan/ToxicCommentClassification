import re
import time
import string
import random
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup


def clean_text(text: str) -> str:
    """
    Cleans text by removing redundant parts.
    :param text: sentence
    :return: cleaned sentence
    """
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    text = ' '.join(text.split())
    return text


def preprocessing_for_bert(data: pd.Series, tokenizer: BertTokenizer) -> Tuple[torch.tensor, torch.tensor]:
    """
    Tokenizes sentences
    :param data: sentences
    :param tokenizer: function that tokenizes single sentence
    :return: token ids and attention masks for all sentences
    """
    input_ids = []
    attention_masks = []

    for sent in tqdm_notebook(data):
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded_sent['input_ids'])
        attention_masks.append(encoded_sent['attention_mask'])

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


class BertClassifier(nn.Module):
    """
    Model for multi classification
    """

    def __init__(self, freeze_bert=True):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 6
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out))
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]  # out['pooler_output']
        logits = self.classifier(last_hidden_state_cls)
        return logits


def initialize_model(device: str, n_batches: int, epochs: int = 4) -> Tuple[
    BertClassifier, AdamW, get_linear_schedule_with_warmup]:
    """
    Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    :param device: device for model training (cuda or cpu)
    :param n_batches: number of batches
    :param epochs: number of epochs to train
    :return: model, optimizer, scheduler
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)

    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,  # Default learning rate
                      eps=1e-8  # Default epsilon value
                      )

    # Total number of training steps
    total_steps = n_batches * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


def set_seed(seed_value: int = 42) -> None:
    """
    Sets seed for main modules
    :param seed_value: the value of seed
    :return: None
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(model: BertClassifier, train_dataloader: DataLoader, optimizer: AdamW,
          scheduler: get_linear_schedule_with_warmup, loss_fn: nn.BCEWithLogitsLoss, device: str,
          val_dataloader: DataLoader = None, epochs: int = 4, evaluation: bool = False) -> None:
    """
    Main function to train the model and evaluate on validation set
    :param model: bert model
    :param train_dataloader: training data
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param loss_fn: loss function
    :param device: device for model training (cuda or cpu)
    :param val_dataloader: validation data
    :param epochs: number of epochs
    :param evaluation: whether to perform evaluation
    :return: None
    """
    print("Start training...\n")
    for epoch_i in range(epochs):
        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()

        for step, batch in enumerate(tqdm_notebook(train_dataloader)):
            if step == 0:
                print(
                    f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Mean ROCAUC':^9} | {'Elapsed':^9}")
                print("-" * 70)
            batch_counts += 1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits, b_labels.float())
            batch_loss += loss.item()
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step % 10 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {'-':^9} | {time_elapsed:^9.2f}")
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

    avg_train_loss = total_loss / len(train_dataloader)

    print("-" * 70)

    if evaluation:
        val_loss, val_accuracy, rocauc_mean_score = evaluate(model, val_dataloader, loss_fn, device)
        time_elapsed = time.time() - t0_epoch
        print(
            f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {rocauc_mean_score:^9.2f} | {time_elapsed:^9.2f}")
        print("-" * 70)
    print("\n")

    print("Training complete!")


def evaluate(model: BertClassifier, val_dataloader: DataLoader, loss_fn: nn.BCEWithLogitsLoss, device: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates model performance by calculating Loss value, Accuracy score, and ROC-AUC score.
    :param model: bert model
    :param val_dataloader: validation data
    :param loss_fn: loss function
    :param device: device for model training (cuda or cpu)
    :return: loss, accuracy, ROC-AUC score
    """
    model.eval()
    val_accuracy = []
    val_loss = []
    preds = []
    true_labels = []
    for batch in val_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        loss = loss_fn(logits, b_labels.float())
        val_loss.append(loss.item())
        accuracy = accuracy_thresh(logits.view(-1, 6), b_labels.view(-1, 6))
        val_accuracy.append(accuracy)

        # Roc Auc
        y_pred = logits.view(-1, 6).sigmoid()
        y_pred = y_pred.cpu().detach().numpy()
        y_true = b_labels.cpu().detach().numpy()
        preds.append(y_pred)
        true_labels.append(y_true)

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    preds = np.concatenate(preds)
    true_labels = np.concatenate(true_labels)
    rocauc_mean_score = rocauc_score(preds, true_labels)

    return val_loss, val_accuracy, rocauc_mean_score


def accuracy_thresh(y_pred: torch.tensor, y_true: torch.tensor, thresh: float = 0.5, sigmoid: bool = True) -> float:
    """
    Calculates accuracy score
    :param y_pred: predicted values
    :param y_true: actual values
    :param thresh: threshold
    :param sigmoid: whether to apply sigmoid function
    :return: average accuracy score
    """
    if sigmoid:
        y_pred = y_pred.sigmoid()
    return ((y_pred > thresh) == y_true.byte()).float().mean().item()


def rocauc_score(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Calculates average ROC-AUC score
    :param y_pred: predicted values
    :param y_true: actual values
    :return: average ROC-AUC score
    """
    scores = []
    for dim in range(y_pred.shape[1]):
        scores.append(roc_auc_score(y_true[:, dim], y_pred[:, dim]))
    return np.mean(scores)
