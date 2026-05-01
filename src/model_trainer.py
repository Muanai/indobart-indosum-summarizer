from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback


class SummarizationTrainer:
    def __init__(self, model, tokenizer, tokenized_datasets, compute_metrics_fn):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = tokenized_datasets["train"]
        self.eval_dataset = tokenized_datasets["validation"]
        self.compute_metrics_fn = compute_metrics_fn

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )

    def _get_training_args(self):
        return Seq2SeqTrainingArguments(
            output_dir="./indosum-bart-summarization",
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="rouge2",
            greater_is_better=True,
            logging_strategy="steps",
            logging_steps=10,
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            load_best_model_at_end=True,
            save_total_limit=2,
            num_train_epochs=20,
            predict_with_generate=True,
            fp16=True,
            report_to="none"
        )

    def execute_training(self):
        training_args = self._get_training_args()

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
        )

        trainer.train()