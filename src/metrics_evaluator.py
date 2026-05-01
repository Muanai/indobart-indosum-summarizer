import numpy as np
import evaluate


class RougeEvaluator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge = evaluate.load("rouge")

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}