from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


class DataPreprocessor:
    def __init__(self, tokenizer, max_input_length=1024, max_target_length=128):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def split_dataset(self, df_clean):
        train_val, test_df = train_test_split(df_clean, test_size=0.1, random_state=42)
        train_df, val_df = train_test_split(train_val, test_size=0.1111, random_state=42)

        return DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(val_df),
            'test': Dataset.from_pandas(test_df)
        })

    def _tokenize_fn(self, examples):
        inputs = [ex for ex in examples["text"]]
        targets = [ex for ex in examples["target"]]

        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def execute_tokenization(self, raw_datasets):
        return raw_datasets.map(self._tokenize_fn, batched=True)