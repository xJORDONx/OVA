import pandas as pd
import json

# Load dataset
train_df = pd.read_csv("C:/Users/vishu/swda-eot-bc-balanced_train.csv")
test_df = pd.read_csv("C:/Users/vishu/swda-eot-bc-balanced_test.csv")
validation_df = pd.read_csv("C:/Users/vishu/swda-eot-bc-balanced_validation.csv")

def process_conversations(df):
    """
    Convert dialogue data into a structured format for fine-tuning.
    """
    conversations = []
    for i in range(len(df) - 1):
        user_utterance = df.iloc[i]['text']
        bot_response = df.iloc[i + 1]['text']
        conversations.append({"prompt": user_utterance, "response": bot_response})
    return conversations

# Process and save
dataset_splits = {
    "train": process_conversations(train_df),
    "validation": process_conversations(validation_df),
    "test": process_conversations(test_df)
}

with open("processed_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset_splits, f, indent=4)

print("✅ Data preprocessing complete! Saved as 'processed_dataset.json'")
