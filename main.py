import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from huggingface_hub import login
from torch.nn import CrossEntropyLoss


f = open("llama_token.txt", "r")
llama_token = f.read()

login(llama_token)

training_set_path = "training_set/sms_spam_train_modified.csv"
df = pd.read_csv(training_set_path, encoding="ISO-8859-9")

df["Label"] = df["Label"].map({"ham": 0, "spam": 1})
dataset = Dataset.from_pandas(df)

model_name = "meta-llama/Llama-2-7b-chat-hf"

# Calling the pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(df):
    tokens = tokenizer(df["Message"], padding="max_length", truncation=True, max_length=128)
    tokens["labels"] = df["Label"]  # Ensure labels are included
    return tokens

# Apply tokenization (batched=true takes inputs as a batch, which is more efficient)
tokenized_dataset = dataset.map(tokenize, batched=True)

# Delete raw data
tokenized_dataset = tokenized_dataset.remove_columns(["Message"])

# Y value must be named "labels" for the model to understand 
tokenized_dataset = tokenized_dataset.rename_column("Label", "labels")  

split_data = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_data["train"]
eval_dataset = split_data["test"]


# Load LLaMA 2 model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama2_spam",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    push_to_hub=False
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Start Training
trainer.train()

metrics = trainer.evaluate()
print(metrics)

# Save model and tokenizer
model.save_pretrained("./llama2_spam_model")
tokenizer.save_pretrained("./llama2_spam_model")
