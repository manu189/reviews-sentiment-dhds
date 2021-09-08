import re
import string

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def to_lower(text):
    return text.lower()

def remove_digits(text):
    pattern = r'[0-9]'
    # Match all digits in the string and replace them with an empty string
    return re.sub(pattern, '', text)

def remove_exclamation_question_marks(text):
    text = text.replace('!', '')
    return text.replace('?', '')

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def clean_text(text):
    text = remove_emoji(text)
    text = remove_digits(text)
    text = remove_exclamation_question_marks(text)
    text = remove_punctuation(text)
    return to_lower(text)

    