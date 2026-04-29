import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def smart_join(data):
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return " ".join([smart_join(item) for item in data if item is not None])
    return str(data)