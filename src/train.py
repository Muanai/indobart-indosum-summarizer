from transformers import MBartForConditionalGeneration
from indobenchmark import IndoNLGTokenizer

from src.data_manager import IndoSumManager
from src.utils.patcher import apply_tokenizer_patches
from src.utils.generation_configurator import apply_custom_generation_config
from src.tokenization_manager import DataPreprocessor
from src.metrics_evaluator import RougeEvaluator
from src.model_trainer import SummarizationTrainer


def main():
    base_path = "/kaggle/input/datasets/muanaikhalifah/indosum/"
    model_checkpoint = "indobenchmark/indobart-v2"

    data_manager = IndoSumManager(base_path)
    dataset, max_input, max_target = data_manager.run_pipeline()

    df_clean = dataset['train'].to_pandas()

    raw_tokenizer = IndoNLGTokenizer.from_pretrained(model_checkpoint)
    tokenizer = apply_tokenizer_patches(raw_tokenizer)

    raw_model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)
    model = apply_custom_generation_config(raw_model, tokenizer)

    preprocessor = DataPreprocessor(
        tokenizer=tokenizer,
        max_input_length=max_input,
        max_target_length=max_target
    )

    raw_datasets = preprocessor.split_dataset(df_clean)
    tokenized_datasets = preprocessor.execute_tokenization(raw_datasets)

    evaluator = RougeEvaluator(tokenizer=tokenizer)

    trainer_manager = SummarizationTrainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_datasets=tokenized_datasets,
        compute_metrics_fn=evaluator.compute_metrics
    )

    trainer_manager.execute_training()


if __name__ == "__main__":
    main()