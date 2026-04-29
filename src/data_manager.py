import glob
import pandas as pd
from datasets import load_dataset, Dataset
from utils.cleaning import clean_text, smart_join


class IndoSumManager:
    def __init__(self, base_path):
        self.base_path = base_path

    def _load_data(self):
        train_files = sorted(glob.glob(self.base_path + "train*.jsonl"))
        val_files = sorted(glob.glob(self.base_path + "dev*.jsonl"))
        test_files = sorted(glob.glob(self.base_path + "test*.jsonl"))

        data_files = {
            "train": train_files,
            "validation": val_files,
            "test": test_files
        }
        return load_dataset("json", data_files=data_files)

    def _flatten_fn(self, example):
        return {
            "text": smart_join(example["paragraphs"]),
            "target": smart_join(example["summary"])
        }

    def _clean_fn(self, example):
        return {
            "text": clean_text(example["text"]),
            "target": clean_text(example["target"])
        }

    def run_pipeline(self):
        dataset = self._load_data()

        # Flattening
        cols = dataset['train'].column_names
        dataset = dataset.map(self._flatten_fn, remove_columns=cols)

        # Deduplication
        df_train = pd.DataFrame(dataset['train'])
        init_len = len(df_train)
        df_clean = df_train.drop_duplicates(subset=['text']).reset_index(drop=True)
        dataset['train'] = Dataset.from_pandas(df_clean)

        # Cleaning
        dataset = dataset.map(self._clean_fn)

        # Stats Calculation
        df_stats = pd.DataFrame(dataset['train'])
        df_stats['article_len'] = df_stats['text'].apply(lambda x: len(str(x).split()))
        df_stats['summary_len'] = df_stats['target'].apply(lambda x: len(str(x).split()))

        p95_art = df_stats['article_len'].quantile(0.95)
        p95_sum = df_stats['summary_len'].quantile(0.95)

        self._print_report(init_len, len(df_clean), p95_art, p95_sum)

        return dataset, int(p95_art * 1.3), int(p95_sum * 1.3)

    def _print_report(self, old, new, p95_a, p95_s):
        print("=" * 30 + "\nDATA REPORT\n" + "=" * 30)
        print(f"Removed: {old - new} duplicates")
        print(f"P95 Article: {p95_a:.2f}")
        print(f"P95 Summary: {p95_s:.2f}")