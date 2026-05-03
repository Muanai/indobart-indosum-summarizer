from transformers import GenerationConfig


def apply_custom_generation_config(model, tokenizer):
    id_id_token = tokenizer.convert_tokens_to_ids("[ind]")

    if id_id_token == tokenizer.unk_token_id:
        id_id_token = tokenizer.bos_token_id

    generation_config = GenerationConfig()
    generation_config.max_length = 128
    generation_config.min_length = 30
    generation_config.num_beams = 4
    generation_config.early_stopping = True
    generation_config.no_repeat_ngram_size = 3

    generation_config.forced_bos_token_id = id_id_token
    generation_config.decoder_start_token_id = id_id_token
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    model.generation_config = generation_config

    return model