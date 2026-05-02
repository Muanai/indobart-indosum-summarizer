import types

def apply_tokenizer_patches(tokenizer):
    _original_pad = tokenizer.pad

    def custom_pad(*args, **kwargs):
        kwargs.pop('padding_side', None)
        return _original_pad(*args, **kwargs)

    tokenizer.pad = custom_pad

    def custom_save_vocabulary(self, save_directory, filename_prefix=None):
        return ()

    tokenizer.save_vocabulary = types.MethodType(custom_save_vocabulary, tokenizer)

    _original_decode = tokenizer.decode

    def custom_decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=None, **kwargs):
        return _original_decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)

    tokenizer.decode = types.MethodType(custom_decode, tokenizer)

    return tokenizer